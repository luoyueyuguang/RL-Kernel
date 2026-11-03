# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the fused routed Q/K norm + RoPE against the eager composition.

The fused path is one kernel per (token, head) row: the RMS statistic, the
per-expert weight, the declared-dtype rounding and the rotation all happen in a
single pass over the row, and ``q``/``k`` are launched from one wrapper.

The baseline is the composition the model ran before fusion: a masked
``torch.nn.functional.rms_norm`` per expert (two calls, one per selected weight)
followed by ``NativeRoPEOp``, whose rotation is a broadcast-multiply over a
duplicated FP32 cos/sin table built on every call.

Usage:
    python benchmarks/backends/cuda/benchmark_mot_qk_norm_rope.py
    python benchmarks/backends/cuda/benchmark_mot_qk_norm_rope.py \
        --rows 4096 --n-heads 28 --n-kv-heads 4 --dtype bf16
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn.functional as F
from tabulate import tabulate

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.reference.rope.mot_qk_norm_rope import HEAD_DIM, build_rope_table  # noqa: E402
from rl_engine.reference.rope.rope import NativeRoPEOp  # noqa: E402

_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _parse_dtype(value: str) -> torch.dtype:
    try:
        return _DTYPES[value.lower()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {value}") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--rows", type=int, default=4096, help="token rows in the call")
    parser.add_argument("--n-heads", type=int, default=28)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def _make_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if args.head_dim != HEAD_DIM:
        raise SystemExit(f"this row is pinned to head_dim={HEAD_DIM}")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    weights = [
        torch.randn((args.head_dim,), generator=generator, device=device, dtype=args.dtype)
        for _ in range(4)
    ]
    cos, sin = build_rope_table(args.rows, args.head_dim, device=device)
    return {
        "q": torch.randn(
            (args.rows, args.n_heads, args.head_dim),
            generator=generator,
            device=device,
            dtype=args.dtype,
        ),
        "k": torch.randn(
            (args.rows, args.n_kv_heads, args.head_dim),
            generator=generator,
            device=device,
            dtype=args.dtype,
        ),
        "q_weight_und": weights[0],
        "q_weight_gen": weights[1],
        "k_weight_und": weights[2],
        "k_weight_gen": weights[3],
        "gen_mask": torch.arange(args.rows, device=device) % 2 == 1,
        "positions": torch.arange(args.rows, device=device, dtype=torch.long),
        "cos": cos,
        "sin": sin,
        "q_eps": 1e-6,
        "k_eps": 1e-6,
    }


def _unsqueeze(inputs: dict[str, object]) -> dict[str, object]:
    names = ("q", "k", "q_weight_und", "q_weight_gen", "k_weight_und", "k_weight_gen")
    args = dict(inputs)
    for name in names:
        args[name] = inputs[name].detach().clone().requires_grad_(True)
    return args


def _fused_forward(op, inputs: dict[str, object]) -> None:
    op.forward(**inputs)


def _fused_forward_backward(op, inputs: dict[str, object]) -> None:
    args = _unsqueeze(inputs)
    q_out, k_out = op.forward(**args)
    torch.autograd.grad(
        (q_out, k_out),
        [args[name] for name in ("q", "k")],
        grad_outputs=(torch.ones_like(q_out), torch.ones_like(k_out)),
    )


def _eager_half(
    x: torch.Tensor,
    weight_und: torch.Tensor,
    weight_gen: torch.Tensor,
    gen_mask: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Per-expert RMSNorm (two masked calls) + the reference rotation."""

    rows, heads, dim = x.shape
    flat = x.reshape(-1, dim)
    selected = gen_mask[:, None].expand(-1, heads).reshape(-1)
    normalized = torch.empty_like(flat)
    for expert_rows, weight in ((~selected, weight_und), (selected, weight_gen)):
        if not bool(expert_rows.any()):
            continue
        index = expert_rows.nonzero(as_tuple=True)[0]
        normalized.index_copy_(
            0, index, F.rms_norm(flat.index_select(0, index), (dim,), weight, float(eps))
        )
    # NativeRoPEOp builds a [1, 1, S, D] cos/sin from the [S] positions, so the
    # rotation runs on the head-major [H, S, D] view (broadcast to [1, H, S, D]).
    head_major = normalized.reshape(rows, heads, dim).permute(1, 0, 2).contiguous()
    rotated = NativeRoPEOp().forward(head_major, positions).reshape(heads, rows, dim)
    return rotated.permute(1, 0, 2)


def _eager_forward(inputs: dict[str, object]) -> None:
    for prefix, eps_key in (("q", "q_eps"), ("k", "k_eps")):
        _eager_half(
            inputs[prefix],
            inputs[f"{prefix}_weight_und"],
            inputs[f"{prefix}_weight_gen"],
            inputs["gen_mask"],
            inputs["positions"],
            inputs[eps_key],
        )


def _eager_forward_backward(inputs: dict[str, object]) -> None:
    args = _unsqueeze(inputs)
    outputs = []
    for prefix, eps_key in (("q", "q_eps"), ("k", "k_eps")):
        value = _eager_half(
            args[prefix],
            args[f"{prefix}_weight_und"],
            args[f"{prefix}_weight_gen"],
            args["gen_mask"],
            args["positions"],
            args[eps_key],
        )
        outputs.append(value)
    torch.autograd.grad(
        tuple(outputs),
        [args[name] for name in ("q", "k")],
        grad_outputs=tuple(torch.ones_like(value) for value in outputs),
    )


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_mb(fn, warmup: int = 3, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**2)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("the fused mot_qk_norm_rope benchmark needs a GPU")
    device = torch.device("cuda")
    from rl_engine.backends.cuda.rope.mot_qk_norm_rope import MotQkNormRopeCudaOp

    fused = MotQkNormRopeCudaOp()
    inputs = _make_inputs(args, device)

    rows = []
    for label, fused_fn, eager_fn in (
        ("forward", _fused_forward, _eager_forward),
        ("forward+backward", _fused_forward_backward, _eager_forward_backward),
    ):
        fused_ms = _time_ms(lambda: fused_fn(fused, inputs), args.warmup, args.iters)
        eager_ms = _time_ms(lambda: eager_fn(inputs), args.warmup, args.iters)
        fused_peak = _peak_mb(lambda: fused_fn(fused, inputs))
        eager_peak = _peak_mb(lambda: eager_fn(inputs))
        rows.append(
            {
                "phase": label,
                "fused_ms": round(fused_ms, 4),
                "unfused_ms": round(eager_ms, 4),
                "speedup": round(eager_ms / fused_ms, 2),
                "fused_peak_mb": round(fused_peak, 1),
                "unfused_peak_mb": round(eager_peak, 1),
            }
        )

    header = (
        f"mot_qk_norm_rope on {torch.cuda.get_device_name(device)} "
        f"(dtype={args.dtype}, rows={args.rows}, Hq={args.n_heads}, "
        f"Hk={args.n_kv_heads}, head_dim={args.head_dim})"
    )
    print(header)
    print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    main()
