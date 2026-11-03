# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent FP32/FP64 reference for the BAGEL MoT routed Q/K norm + RoPE row.

BAGEL-7B-MoT runs two expert streams (text and image tokens) through one attention
block. Each stream has its own per-head RMSNorm weight for q and for k, and the
normalized heads are then rotated with NeoX (rotate-half) RoPE:

    rstd[t]   = 1 / sqrt(mid_split_mean(x[t]^2) + eps)          # per row = per head
    normed[t] = ((f32(x[t]) * rstd[t]) * w_e[t])                # w_e = gen ? w_gen : w_und
    y_lo[t]   = normed[t, :64] * cos - normed[t, 64:] * sin
    y_hi[t]   = normed[t, 64:] * cos + normed[t, :64] * sin

The reduction tree, the multiplication order and the single rounding at the
declared dtype boundary are the operator's contract (``mot-qk-norm-rope-v1``);
the CUDA and Triton backends reproduce them op for op, so their outputs are
byte-identical to each other and to this reference's declared-dtype path.
"""

from __future__ import annotations

import torch
from torch import Tensor

MOT_QK_NORM_ROPE_CONTRACT = "mot-qk-norm-rope-v1"
HEAD_DIM = 128
HALF = 64


def build_rope_table(
    max_positions: int,
    head_dim: int,
    *,
    device: torch.device | str,
    theta: float = 1e6,
) -> tuple[Tensor, Tensor]:
    """FP32 NeoX cos/sin tables of shape ``[max_positions, head_dim // 2]``.

    Same arithmetic as ``NativeRoPEOp._compute_cos_sin``: the table stores one
    half of the duplicated embedding, so ``cos[pos]`` multiplies both halves.
    """

    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(max_positions, dtype=torch.float32, device=device).unsqueeze(-1)
    freqs = pos * inv_freq
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _rstd_mid_split(x32: Tensor, eps: float) -> Tensor:
    """Per-row rstd over the contiguous mid-split tree ``T(l, r) = T(l, m) + T(m, r)``.

    Seven adjacent-pair levels reduce ``[..., 128]`` to ``[..., 1]``: the level
    sums are the same association a warp butterfly plus a four-warp combine
    produces in the kernels, and the mean divides the *complete* tree.
    """

    s = x32.square()
    for _ in range(7):
        s = s.reshape(*s.shape[:-1], s.shape[-1] // 2, 2).sum(-1)
    return 1.0 / torch.sqrt(s / float(HEAD_DIM) + eps)


def _apply(
    x: Tensor,
    weight_und: Tensor,
    weight_gen: Tensor,
    gen_mask: Tensor,
    positions: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float,
    *,
    output_dtype: torch.dtype,
    fp32: bool,
    fp64: bool,
) -> Tensor:
    """Shared arithmetic of the three entry points; the only difference is rounding."""

    compute_dtype = torch.float64 if fp64 else torch.float32
    x32 = x.to(compute_dtype)
    w = torch.where(
        gen_mask[:, None],
        weight_gen.to(compute_dtype),
        weight_und.to(compute_dtype),
    )  # [N, D]
    rstd = _rstd_mid_split(x32, eps)  # [..., 1]: the tree collapses the head dim
    normed = (x32 * rstd) * w.unsqueeze(1)  # w broadcasts over heads
    if not (fp32 or fp64):
        # The declared dtype boundary: one RNE to the caller's dtype before the
        # rotation, exactly as the kernels round their normalized values.
        normed = normed.to(x.dtype).to(compute_dtype)
    lo, hi = normed[..., :HALF], normed[..., HALF:]
    c = cos[positions].to(compute_dtype).unsqueeze(1)  # [N, 1, HALF]
    s = sin[positions].to(compute_dtype).unsqueeze(1)
    out_lo = lo * c - hi * s
    out_hi = hi * c + lo * s
    out = torch.cat([out_lo, out_hi], dim=-1)
    return out if (fp32 or fp64) else out.to(output_dtype)


def _check_inputs(
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
) -> None:
    for name, value in (("q", q), ("k", k)):
        if value.dim() != 3:
            raise ValueError(f"{name} must be 3-D [N, H, {HEAD_DIM}], got {value.dim()}-D")
        if value.shape[-1] != HEAD_DIM:
            raise ValueError(f"{name}.shape[-1] must be {HEAD_DIM}, got {value.shape[-1]}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"q and k must share the row count, got {q.shape[0]} and {k.shape[0]}")
    rows = q.shape[0]
    if gen_mask.shape != (rows,):
        raise ValueError(f"gen_mask must be [{rows}], got {tuple(gen_mask.shape)}")
    if gen_mask.dtype is not torch.bool:
        raise ValueError(f"gen_mask must be bool, got {gen_mask.dtype}")
    if positions.shape != (rows,):
        raise ValueError(f"positions must be [{rows}], got {tuple(positions.shape)}")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"positions must be an integer tensor, got {positions.dtype}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must share a shape, got {cos.shape} and {sin.shape}")
    if cos.dim() != 2 or cos.shape[1] != HALF:
        raise ValueError(f"cos/sin must be [P, {HALF}], got {tuple(cos.shape)}")
    if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
        raise ValueError(f"cos/sin must be float32, got {cos.dtype} and {sin.dtype}")
    for name, weight in (
        ("q_weight_und", q_weight_und),
        ("q_weight_gen", q_weight_gen),
        ("k_weight_und", k_weight_und),
        ("k_weight_gen", k_weight_gen),
    ):
        if tuple(weight.shape) != (HEAD_DIM,):
            raise ValueError(f"{name} must be [{HEAD_DIM}], got {tuple(weight.shape)}")


class NativeMotQkNormRopeOp:
    """Pure PyTorch reference of the routed per-head Q/K RMSNorm + NeoX RoPE.

    The two expert weights are selected per row by ``gen_mask`` (True = the
    generator stream, False = the understanding stream). ``forward`` is the
    declared-dtype path, ``forward_fp32`` is the ground truth the GPU backends are
    compared against and ``forward_fp64`` re-runs the same declared tree in FP64
    so the FP32 tree itself can be checked on a CPU-only host.
    """

    op_class = "reduction"
    is_batch_invariant = True
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
        return self._run(
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
            mode="declared",
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
        return self._run(
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
            mode="fp32",
        )

    def forward_fp64(
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
        return self._run(
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
            mode="fp64",
        )

    def _run(
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
        q_eps: float,
        k_eps: float,
        *,
        mode: str,
    ) -> tuple[Tensor, Tensor]:
        _check_inputs(
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
        )
        fp32 = mode == "fp32"
        fp64 = mode == "fp64"
        if mode not in ("declared", "fp32", "fp64"):
            raise ValueError(f"unknown reference mode {mode!r}")
        outs = []
        for value, w_und, w_gen, eps in (
            (q, q_weight_und, q_weight_gen, q_eps),
            (k, k_weight_und, k_weight_gen, k_eps),
        ):
            if mode == "declared":
                output_dtype = value.dtype
            else:
                output_dtype = torch.float64 if fp64 else torch.float32
            outs.append(
                _apply(
                    value,
                    w_und,
                    w_gen,
                    gen_mask,
                    positions,
                    cos,
                    sin,
                    float(eps),
                    output_dtype=output_dtype,
                    fp32=fp32,
                    fp64=fp64,
                )
            )
        return outs[0], outs[1]


__all__ = [
    "HALF",
    "HEAD_DIM",
    "MOT_QK_NORM_ROPE_CONTRACT",
    "NativeMotQkNormRopeOp",
    "build_rope_table",
]
