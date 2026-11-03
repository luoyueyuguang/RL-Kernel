# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA backend for the BAGEL MoT routed Q/K RMSNorm + NeoX RoPE row.

One kernel serves both tensors: a block per ``(token, head)`` row of 128 elements
reduces ``x^2`` over the declared contiguous mid-split tree (a warp butterfly
with ascending lane offsets 1, 2, 4, 8, 16 plus a four-warp combine), forms
``rstd = 1 / sqrt(mean + eps)``, multiplies by the row's expert weight, rounds the
normalized value once to the declared dtype, and rotates the two halves in that
declared dtype with the table cast to it (RFC #435).

The `.cu` is plain CUDA C, so ``hipcc`` builds the same source for ROCm and the
two kernels are byte-identical. The backward is deterministic by construction:
``dx`` is row-local (the same mid-split tree), and the per-expert ``dW`` is a
per-row FP32 contributions followed by an ascending-row left fold.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32
from rl_engine.reference.rope.mot_qk_norm_rope import (
    HALF,
    HEAD_DIM,
    MOT_QK_NORM_ROPE_CONTRACT,
    NativeMotQkNormRopeOp,
)

REQUIRED_SYMBOLS = ("mot_qk_norm_rope_forward", "mot_qk_norm_rope_backward")


def _kernel_available() -> bool:
    return _EXT_AVAILABLE and all(hasattr(_C, name) for name in REQUIRED_SYMBOLS)


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
    """Whether the fused kernel can serve these operands without a cast or copy.

    Everything the kernel indexes is required: one CUDA device, ``[N, H, 128]``
    unit-inner-stride contiguous ``q``/``k``, ``[128]`` contiguous weights in the
    caller's dtype, an FP32 ``[P, 64]`` table, a bool mask and int64 positions.
    """

    if not _kernel_available():
        return False
    if not (q.is_cuda and k.is_cuda):
        return False
    device = q.device
    tensors = (
        k,
        q_weight_und,
        q_weight_gen,
        k_weight_und,
        k_weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
    )
    if any(value.device != device for value in tensors):
        return False
    if q.dim() != 3 or k.dim() != 3:
        return False
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
        return False
    if q.shape[0] != k.shape[0]:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    if not (q.is_contiguous() and k.is_contiguous()):
        return False
    dtype = q.dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if k.dtype != dtype:
        return False
    for weight in (q_weight_und, q_weight_gen, k_weight_und, k_weight_gen):
        if weight.dim() != 1 or weight.shape[0] != HEAD_DIM:
            return False
        if weight.dtype != dtype or not weight.is_contiguous():
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
    if not positions.is_contiguous():
        return False
    return True


def _reduce_weight_rows(rows: Tensor, mask: Tensor, dtype: torch.dtype) -> Tensor:
    """FP32 ascending-row left fold of the masked per-row contributions."""

    masked = rows * mask.to(dtype=rows.dtype).unsqueeze(-1)
    if torch.version.hip is None:
        reduced = reduce_rows_fp32(masked)
    else:
        # reduce_rows_fp32's CUDA kernel does not exist in the ROCm build; the
        # Triton fold walks the same ascending row order.
        from rl_engine.backends.shared.triton.rope.mot_qk_norm_rope import (
            mot_qk_norm_rope_fold_rows,
        )

        reduced = mot_qk_norm_rope_fold_rows(masked.contiguous())
    return reduced.to(dtype=dtype)


class _MotQkNormRopeFunction(torch.autograd.Function):
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
        q_out, k_out, q_rstd, k_rstd = _C.mot_qk_norm_rope_forward(
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
            float(q_eps),
            float(k_eps),
            True,
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

        dq, dk, q_contrib, k_contrib = _C.mot_qk_norm_rope_backward(
            grad_q.contiguous(),
            grad_k.contiguous(),
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

        q_rows = q_contrib.view(-1, HEAD_DIM)
        k_rows = k_contrib.view(-1, HEAD_DIM)
        q_expert = gen_mask.repeat_interleave(q.shape[1])
        k_expert = gen_mask.repeat_interleave(k.shape[1])
        # Two folds per tensor: one per expert. A mask that leaves an expert
        # unused produces the zero vector, which is that expert's exact gradient.
        dq_weight_und = _reduce_weight_rows(q_rows, ~q_expert, q_weight_und.dtype)
        dq_weight_gen = _reduce_weight_rows(q_rows, q_expert, q_weight_gen.dtype)
        dk_weight_und = _reduce_weight_rows(k_rows, ~k_expert, k_weight_und.dtype)
        dk_weight_gen = _reduce_weight_rows(k_rows, k_expert, k_weight_gen.dtype)

        record_backward(
            "mot_qk_norm_rope",
            kernel_id=(
                "rl_engine._C.mot_qk_norm_rope_backward"
                "+rl_engine.ops.autograd.vjp_fp32.reduce_rows_fp32"
            ),
            impl=MotQkNormRopeCudaOp.backward_impl,
            family="cuda",
        )

        return (
            dq,
            dk,
            dq_weight_und,
            dq_weight_gen,
            dk_weight_und,
            dk_weight_gen,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MotQkNormRopeCudaOp:
    """Fused routed Q/K RMSNorm + RoPE on the precompiled CUDA (or hipified) kernel."""

    op_class = "reduction"
    is_batch_invariant = True
    backend_id = "rlkernel.cuda.mot_qk_norm_rope"
    backward_impl = "cuda_mot_qk_norm_rope_mid_split_dx_rowfold_dw"
    contract = MOT_QK_NORM_ROPE_CONTRACT

    def __init__(self) -> None:
        if not _kernel_available():
            raise RuntimeError("mot_qk_norm_rope CUDA kernel is unavailable; rebuild rl_engine._C")

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
            q_out, k_out, _q_rstd, _k_rstd = _C.mot_qk_norm_rope_forward(
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
                float(q_eps),
                float(k_eps),
                False,
            )
            return q_out, k_out
        return _MotQkNormRopeFunction.apply(
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


__all__ = ["MotQkNormRopeCudaOp", "supports"]
