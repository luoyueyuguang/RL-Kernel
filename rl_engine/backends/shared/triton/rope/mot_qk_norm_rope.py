# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton backend for the BAGEL MoT routed Q/K RMSNorm + NeoX RoPE row.

Same contract as the CUDA source (``mot-qk-norm-rope-v1``), row for row: one
program per ``(token, head)`` row of 128 elements, the ``x^2`` sum taken over the
contiguous mid-split tree (seven adjacent-pair levels), ``rstd = 1 / sqrt(mean +
eps)`` with correctly-rounded division and square root, ``(x * rstd) * w`` with
the row's expert weight, exactly one rounding to the declared dtype, and the
rotation in FP32 with the FP32 cos/sin table.

It is the ROCm candidate for the row (Triton is JIT-compiled per device, so the
same file serves ROCm and CUDA) and the cross-backend parity reference: on one
device the Triton and the hipified CUDA kernels are byte-identical.

``enable_fp_fusion=False`` keeps the compiler from contracting the rotation into
FMA, which is what makes the CUDA and Triton outputs comparable bit for bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.reference.rope.mot_qk_norm_rope import (
    HALF,
    HEAD_DIM,
    MOT_QK_NORM_ROPE_CONTRACT,
    NativeMotQkNormRopeOp,
)

_FOLD_BLOCK = 128


@triton.jit
def _mot_qk_norm_rope_fwd_kernel(
    X,
    W_UND,
    W_GEN,
    GEN,
    POS,
    COS,
    SIN,
    OUT,
    RSTD,
    H: tl.constexpr,
    EPS,
    STORE_RSTD: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    base = (token * H + head) * 128

    gen = tl.load(GEN + token).to(tl.int32) != 0
    d128 = tl.arange(0, 128)
    x = tl.load(X + base + d128).to(tl.float32)

    # The declared contiguous mid-split tree: seven adjacent-pair levels.
    s = x * x
    s = tl.sum(tl.reshape(s, (64, 2)), axis=1)
    s = tl.sum(tl.reshape(s, (32, 2)), axis=1)
    s = tl.sum(tl.reshape(s, (16, 2)), axis=1)
    s = tl.sum(tl.reshape(s, (8, 2)), axis=1)
    s = tl.sum(tl.reshape(s, (4, 2)), axis=1)
    s = tl.sum(tl.reshape(s, (2, 2)), axis=1)
    s = tl.sum(s, axis=0)
    rstd = tl.div_rn(1.0, libdevice.sqrt_rn(s / 128.0 + EPS))
    if STORE_RSTD:
        tl.store(RSTD + token * H + head, rstd)

    d = tl.arange(0, 64)
    lo = tl.load(X + base + d).to(tl.float32)
    hi = tl.load(X + base + d + 64).to(tl.float32)
    w_lo = tl.where(gen, tl.load(W_GEN + d), tl.load(W_UND + d)).to(tl.float32)
    w_hi = tl.where(gen, tl.load(W_GEN + d + 64), tl.load(W_UND + d + 64)).to(tl.float32)
    # The declared dtype boundary: one rounding before the rotation.
    y_lo = ((lo * rstd) * w_lo).to(OUT.dtype.element_ty).to(tl.float32)
    y_hi = ((hi * rstd) * w_hi).to(OUT.dtype.element_ty).to(tl.float32)

    pos = tl.load(POS + token)
    cosine = tl.load(COS + pos * 64 + d)
    sine = tl.load(SIN + pos * 64 + d)
    out_ty = OUT.dtype.element_ty
    tl.store(OUT + base + d, (y_lo * cosine - y_hi * sine).to(out_ty))
    tl.store(OUT + base + d + 64, (y_hi * cosine + y_lo * sine).to(out_ty))


@triton.jit
def _mot_qk_norm_rope_bwd_kernel(
    G,
    X,
    W_UND,
    W_GEN,
    GEN,
    POS,
    COS,
    SIN,
    RSTD,
    DX,
    DW_CONTRIB,
    H: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    row = token * H + head
    base = row * 128

    gen = tl.load(GEN + token).to(tl.int32) != 0
    d = tl.arange(0, 64)
    pos = tl.load(POS + token)
    cosine = tl.load(COS + pos * 64 + d)
    sine = tl.load(SIN + pos * 64 + d)

    g_lo = tl.load(G + base + d).to(tl.float32)
    g_hi = tl.load(G + base + d + 64).to(tl.float32)
    x_lo = tl.load(X + base + d).to(tl.float32)
    x_hi = tl.load(X + base + d + 64).to(tl.float32)
    w_lo = tl.where(gen, tl.load(W_GEN + d), tl.load(W_UND + d)).to(tl.float32)
    w_hi = tl.where(gen, tl.load(W_GEN + d + 64), tl.load(W_UND + d + 64)).to(tl.float32)

    # Inverse rotation through the same declared dtype boundary as the forward.
    dy_lo_rot = g_lo * cosine + g_hi * sine
    dy_hi_rot = g_hi * cosine - g_lo * sine
    dy_lo = dy_lo_rot.to(DX.dtype.element_ty).to(tl.float32)
    dy_hi = dy_hi_rot.to(DX.dtype.element_ty).to(tl.float32)

    contrib_lo = (dy_lo * w_lo) * x_lo
    contrib_hi = (dy_hi * w_hi) * x_hi
    # tree64 over each half, then the pair sum: the same association the forward's
    # seven-level tree takes over the full row.
    acc_lo = tl.sum(tl.reshape(contrib_lo, (32, 2)), axis=1)
    acc_lo = tl.sum(tl.reshape(acc_lo, (16, 2)), axis=1)
    acc_lo = tl.sum(tl.reshape(acc_lo, (8, 2)), axis=1)
    acc_lo = tl.sum(tl.reshape(acc_lo, (4, 2)), axis=1)
    acc_lo = tl.sum(tl.reshape(acc_lo, (2, 2)), axis=1)
    dot_lo = tl.sum(acc_lo, axis=0)
    acc_hi = tl.sum(tl.reshape(contrib_hi, (32, 2)), axis=1)
    acc_hi = tl.sum(tl.reshape(acc_hi, (16, 2)), axis=1)
    acc_hi = tl.sum(tl.reshape(acc_hi, (8, 2)), axis=1)
    acc_hi = tl.sum(tl.reshape(acc_hi, (4, 2)), axis=1)
    acc_hi = tl.sum(tl.reshape(acc_hi, (2, 2)), axis=1)
    dot_hi = tl.sum(acc_hi, axis=0)
    dot = dot_lo + dot_hi

    r = tl.load(RSTD + row)
    coeff = ((((dot * r) * r) * r) / 128.0).to(tl.float32)
    dx_lo = (r * dy_lo) * w_lo - x_lo * coeff
    dx_hi = (r * dy_hi) * w_hi - x_hi * coeff
    dx_ty = DX.dtype.element_ty
    tl.store(DX + base + d, dx_lo.to(dx_ty))
    tl.store(DX + base + d + 64, dx_hi.to(dx_ty))
    tl.store(DW_CONTRIB + base + d, (dy_lo * x_lo) * r)
    tl.store(DW_CONTRIB + base + d + 64, (dy_hi * x_hi) * r)


@triton.jit
def _mot_qk_norm_rope_fold_rows_kernel(
    ROWS,
    OUT,
    N_ROWS,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Ascending-row FP32 left fold, one program over the columns."""

    columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = columns < COLS
    acc = tl.zeros((BLOCK,), tl.float32)
    row = 0
    while row < N_ROWS:
        acc += tl.load(ROWS + row * COLS + columns, mask=active, other=0.0)
        row += 1
    tl.store(OUT + columns, acc, mask=active)


def mot_qk_norm_rope_fold_rows(rows: Tensor) -> Tensor:
    """FP32 ascending-row left fold of ``[N, COLS]`` contributions."""

    if rows.dim() != 2:
        raise ValueError(f"rows must be 2-D [N, COLS], got {tuple(rows.shape)}")
    rows = rows.contiguous()
    columns = rows.shape[1]
    if columns > _FOLD_BLOCK:
        raise ValueError(f"fold supports COLS <= {_FOLD_BLOCK}, got {columns}")
    out = torch.empty((columns,), device=rows.device, dtype=torch.float32)
    _mot_qk_norm_rope_fold_rows_kernel[(1,)](
        rows,
        out,
        rows.shape[0],
        COLS=columns,
        BLOCK=_FOLD_BLOCK,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


def _launch_forward(
    x: Tensor,
    weight_und: Tensor,
    weight_gen: Tensor,
    gen_mask: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float,
    *,
    store_rstd: bool,
    num_warps: int = 1,
) -> tuple[Tensor, Tensor]:
    out = torch.empty_like(x)
    rows = x.shape[0] * x.shape[1]
    # The buffer is write-only and untouched when store_rstd is False: the
    # inference path must not pay for it.
    rstd = (
        torch.empty((rows,), device=x.device, dtype=torch.float32)
        if store_rstd
        else torch.empty((0,), device=x.device, dtype=torch.float32)
    )
    _mot_qk_norm_rope_fwd_kernel[(x.shape[0], x.shape[1])](
        x,
        weight_und,
        weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
        out,
        rstd,
        x.shape[1],
        float(eps),
        STORE_RSTD=store_rstd,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out, rstd


def _launch_backward(
    grad: Tensor,
    x: Tensor,
    weight_und: Tensor,
    weight_gen: Tensor,
    gen_mask: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
    rstd: Tensor,
    *,
    num_warps: int = 1,
) -> tuple[Tensor, Tensor]:
    dx = torch.empty_like(x)
    dw_contrib = torch.empty(
        (x.shape[0] * x.shape[1], HEAD_DIM), device=x.device, dtype=torch.float32
    )
    _mot_qk_norm_rope_bwd_kernel[(x.shape[0], x.shape[1])](
        grad,
        x,
        weight_und,
        weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
        rstd,
        dx,
        dw_contrib,
        x.shape[1],
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return dx, dw_contrib


def supports(
    q: Tensor,
    k: Tensor,
    q_weight_und: Tensor,
    q_weight_gen: Tensor,
    k_weight_und: Tensor,
    k_weight_gen: Tensor,
    gen_mask: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> bool:
    """Whether the Triton kernels can serve these operands without a cast or copy."""

    if not q.is_cuda or not k.is_cuda:
        return False
    device = q.device
    for value in (
        k,
        q_weight_und,
        q_weight_gen,
        k_weight_und,
        k_weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
    ):
        if value.device != device:
            return False
    if q.dim() != 3 or k.dim() != 3:
        return False
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
        return False
    if q.shape[0] != k.shape[0] or q.dtype != k.dtype:
        return False
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    if not (q.is_contiguous() and k.is_contiguous()):
        return False
    for weight in (q_weight_und, q_weight_gen, k_weight_und, k_weight_gen):
        if weight.dim() != 1 or weight.shape[0] != HEAD_DIM:
            return False
        if weight.dtype != q.dtype or not weight.is_contiguous():
            return False
    if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
        return False
    if cos.dim() != 2 or cos.shape != sin.shape or cos.shape[1] != HALF:
        return False
    if not (cos.is_contiguous() and sin.is_contiguous()):
        return False
    rows = q.shape[0]
    if gen_mask.dtype is not torch.bool or tuple(gen_mask.shape) != (rows,):
        return False
    if positions.dtype is not torch.int64 or tuple(positions.shape) != (rows,):
        return False
    return True


class _MotQkNormRopeTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        q_weight_und: Tensor,
        q_weight_gen: Tensor,
        k_weight_und: Tensor,
        k_weight_gen: Tensor,
        gen_mask: Tensor,
        positions: Tensor,
        cos: Tensor,
        sin: Tensor,
        q_eps: float,
        k_eps: float,
    ) -> tuple[Tensor, Tensor]:
        q_out, q_rstd = _launch_forward(
            q,
            q_weight_und,
            q_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            q_eps,
            store_rstd=True,
        )
        k_out, k_rstd = _launch_forward(
            k,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            k_eps,
            store_rstd=True,
        )
        ctx.save_for_backward(
            q,
            k,
            q_weight_und,
            q_weight_gen,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            q_rstd,
            k_rstd,
        )
        ctx.eps = (float(q_eps), float(k_eps))
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: Tensor, grad_k: Tensor):
        (
            q,
            k,
            q_weight_und,
            q_weight_gen,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            q_rstd,
            k_rstd,
        ) = ctx.saved_tensors

        dq, q_contrib = _launch_backward(
            grad_q.contiguous(),
            q,
            q_weight_und,
            q_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            q_rstd,
        )
        dk, k_contrib = _launch_backward(
            grad_k.contiguous(),
            k,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            k_rstd,
        )

        q_expert = gen_mask.repeat_interleave(q.shape[1])
        k_expert = gen_mask.repeat_interleave(k.shape[1])
        dq_weight_und = mot_qk_norm_rope_fold_rows(q_contrib * ~q_expert[:, None])
        dq_weight_gen = mot_qk_norm_rope_fold_rows(q_contrib * q_expert[:, None])
        dk_weight_und = mot_qk_norm_rope_fold_rows(k_contrib * ~k_expert[:, None])
        dk_weight_gen = mot_qk_norm_rope_fold_rows(k_contrib * k_expert[:, None])

        record_backward(
            "mot_qk_norm_rope",
            kernel_id=(
                "rl_engine.backends.shared.triton.rope.mot_qk_norm_rope."
                "_mot_qk_norm_rope_bwd_kernel"
                "+rl_engine.backends.shared.triton.rope.mot_qk_norm_rope."
                "mot_qk_norm_rope_fold_rows"
            ),
            impl=MotQkNormRopeTritonOp.backward_impl,
            family="triton",
        )

        return (
            dq,
            dk,
            dq_weight_und.to(q_weight_und.dtype),
            dq_weight_gen.to(q_weight_gen.dtype),
            dk_weight_und.to(k_weight_und.dtype),
            dk_weight_gen.to(k_weight_gen.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MotQkNormRopeTritonOp:
    """Fused routed Q/K RMSNorm + RoPE in Triton (CUDA and ROCm)."""

    op_class = "reduction"
    is_batch_invariant = True
    backend_id = "rlkernel.triton.mot_qk_norm_rope"
    backward_impl = "triton_mot_qk_norm_rope_mid_split_dx_rowfold_dw"
    contract = MOT_QK_NORM_ROPE_CONTRACT

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        q_weight_und: Tensor,
        q_weight_gen: Tensor,
        k_weight_und: Tensor,
        k_weight_gen: Tensor,
        gen_mask: Tensor,
        positions: Tensor,
        cos: Tensor,
        sin: Tensor,
        q_eps: float = 1e-6,
        k_eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        return self.forward(
            q,
            k,
            q_weight_und,
            q_weight_gen,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            q_eps,
            k_eps,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_weight_und: Tensor,
        q_weight_gen: Tensor,
        k_weight_und: Tensor,
        k_weight_gen: Tensor,
        gen_mask: Tensor,
        positions: Tensor,
        cos: Tensor,
        sin: Tensor,
        q_eps: float = 1e-6,
        k_eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        if not supports(
            q,
            k,
            q_weight_und,
            q_weight_gen,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
        ):
            return NativeMotQkNormRopeOp().forward(
                q,
                k,
                q_weight_und,
                q_weight_gen,
                k_weight_und,
                k_weight_gen,
                gen_mask,
                positions,
                cos,
                sin,
                q_eps,
                k_eps,
            )
        needs_grad = torch.is_grad_enabled() and any(
            value.requires_grad
            for value in (
                q,
                k,
                q_weight_und,
                q_weight_gen,
                k_weight_und,
                k_weight_gen,
            )
        )
        if not needs_grad:
            q_out, _ = _launch_forward(
                q,
                q_weight_und,
                q_weight_gen,
                gen_mask,
                positions,
                cos,
                sin,
                q_eps,
                store_rstd=False,
            )
            k_out, _ = _launch_forward(
                k,
                k_weight_und,
                k_weight_gen,
                gen_mask,
                positions,
                cos,
                sin,
                k_eps,
                store_rstd=False,
            )
            return q_out, k_out
        return _MotQkNormRopeTritonFunction.apply(
            q,
            k,
            q_weight_und,
            q_weight_gen,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            q_eps,
            k_eps,
        )

    def forward_fp32(
        self,
        q: Tensor,
        k: Tensor,
        q_weight_und: Tensor,
        q_weight_gen: Tensor,
        k_weight_und: Tensor,
        k_weight_gen: Tensor,
        gen_mask: Tensor,
        positions: Tensor,
        cos: Tensor,
        sin: Tensor,
        q_eps: float = 1e-6,
        k_eps: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        return NativeMotQkNormRopeOp().forward_fp32(
            q,
            k,
            q_weight_und,
            q_weight_gen,
            k_weight_und,
            k_weight_gen,
            gen_mask,
            positions,
            cos,
            sin,
            q_eps,
            k_eps,
        )


__all__ = [
    "MotQkNormRopeTritonOp",
    "mot_qk_norm_rope_fold_rows",
    "supports",
]
