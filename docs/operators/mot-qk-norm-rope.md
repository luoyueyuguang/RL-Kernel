# Mot QK Norm + RoPE (`mot_qk_norm_rope`)

## Summary

BAGEL-7B-MoT is a Mixture-of-Transformers: the understanding (text) stream and
the generator (image) stream run through one attention block with **two experts'
worth of per-head norm weights**. This operator is the block's Q/K preparation:

```
rstd[t]   = 1 / sqrt(mid_split_mean(x[t]^2) + eps)            # per (token, head) row
normed[t] = ((f32(x[t]) * rstd[t]) * w_e[t])                  # w_e = gen ? w_gen : w_und
y_lo[t]   = round(normed[t, :64]) * cos - round(normed[t, 64:]) * sin
y_hi[t]   = round(normed[t, 64:]) * cos + round(normed[t, :64]) * sin
```

`gen_mask[t]` selects the expert weight **per token row**, so one call carries a
token stream that mixes both modalities — that routing, not the fusion alone, is
what the row exists for. It is the WS1 row of the BAGEL-7B-MoT roadmap
(issue #435) and extends the deterministic RMSNorm (issue #350) and RoPE
(issue #228) rows.

Contract id: `mot-qk-norm-rope-v1`.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("mot_qk_norm_rope", device="cuda")
q_out, k_out = op(q, k, q_weight_und, q_weight_gen, k_weight_und, k_weight_gen,
                  gen_mask, positions, cos, sin, q_eps=1e-6, k_eps=1e-6)
```

Direct construction of a specific backend:

```python
from rl_engine.backends.cuda.rope.mot_qk_norm_rope import MotQkNormRopeCudaOp
from rl_engine.backends.shared.triton.rope.mot_qk_norm_rope import MotQkNormRopeTritonOp
from rl_engine.reference.rope.mot_qk_norm_rope import NativeMotQkNormRopeOp, build_rope_table

cos, sin = build_rope_table(max_positions, 128, device="cuda")   # fp32 [P, 64]
q_out, k_out = MotQkNormRopeCudaOp().forward(
    q, k, q_weight_und, q_weight_gen, k_weight_und, k_weight_gen,
    gen_mask, positions, cos, sin, 1e-6, 1e-6,
)
```

Every backend exposes the same three entry points: `forward` (the declared-dtype
path), `forward_fp32` (the ground-truth path the GPU backends are compared
against) and `forward_fp64` (the reference only: the same declared tree in FP64,
so the FP32 tree itself is checkable on a CPU-only host).

The CUDA and Triton backends are differentiable w.r.t. all six float inputs.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| CUDA | `MotQkNormRopeCudaOp` | `csrc/cuda/rope/mot_qk_norm_rope.cu` → `_C.mot_qk_norm_rope_forward` / `_C.mot_qk_norm_rope_backward` | Plain CUDA C — no PTX, no TMA, no SM90 gate — so it is always compiled. One block (128 threads) per `(token, head)` row. |
| ROCm | `MotQkNormRopeCudaOp` | same source, hipified by the ROCm build | Identical arithmetic to the CUDA backend by construction: the build compiles the *same* `.cu`, so no second implementation exists to drift. Never built with fast math (the source `#error`s if `__USE_FAST_MATH__` is 1). **Not yet measured on ROCm hardware**: the claim is structural, and no AMD host has run this row. |
| Triton | `MotQkNormRopeTritonOp` | `rl_engine/backends/shared/triton/rope/mot_qk_norm_rope.py` | One portable source for CUDA and ROCm (JIT per device), `num_warps=1`, `enable_fp_fusion=False`. The parity reference for the CUDA/ROCm kernel and the fallback when the `_C` symbols are absent. |
| PyTorch | `NativeMotQkNormRopeOp` | `rl_engine/reference/rope/mot_qk_norm_rope.py` | Independent FP32/FP64 reference; the CPU/NPU path and the fallback for any operand the kernels cannot serve. |

## Exact Math and the Frozen Sequences

**Reduction tree.** The RMS statistic is the *contiguous mid-split* tree
`T(l, r) = T(l, m) + T(m, r)` over the 128 head elements, mean taken on the
complete tree:

- Reference: seven adjacent-pair levels — `reshape(..., 64, 2).sum(-1)` down to a
  scalar, then `s / 128 + eps`.
- CUDA: a warp butterfly whose lane offsets ascend (`1, 2, 4, 8, 16`, not the
  reversed order), so lane 0 of warp `w` holds `T(32w..32w+31)`; the four warp
  sums combine as `(p0 + p1) + (p2 + p3)`.
- Triton: the same seven levels, then the last two-level sum.

These are the same association, not merely the same value: every level sums
adjacent pairs of the level below.

**Rounding and rotation.** `rstd = 1.0 / sqrt(mean + eps)` with correctly-rounded
division and square root (`__fdiv_rn`/`__fsqrt_rn` in CUDA, `tl.div_rn`/
`libdevice.sqrt_rn` in Triton) — not `rsqrt`. The Triton spelling matters:
`libdevice.sqrt` and a bare `1.0 / x` lower to approximate instructions on SM90
(184 of 1024 rows differed from `torch.sqrt` by 1 ulp in FP32), which silently
breaks the byte-parity promise below. `sqrt_rn` matches bit for bit.

The normalized value is then rounded **once** to the declared dtype and widened
back for the rotation:

```
y_lo = rne_dtype(normed_lo);  y_hi = rne_dtype(normed_hi)     # the only cast
out_lo = y_lo * c - y_hi * s;  out_hi = y_hi * c + y_lo * s     # fp32, no FMA fusion
out    = rne_dtype(out_lo), rne_dtype(out_hi)
```

Multiplications stay explicitly ordered (`(x * rstd) * w`, `lo*c - hi*s`,
`hi*c + lo*s`); no `torch.addcmul`-style contraction, and Triton launches with
`enable_fp_fusion=False`. cos/sin are the FP32 table the reference builds
(`build_rope_table`, bit-equal to `NativeRoPEOp._compute_cos_sin`'s first half).

**Backward.** RoPE is an orthogonal rotation, so the inverse is the same rotation
with the sine negated:

```
dy_lo = g_lo * c + g_hi * s;   dy_hi = g_hi * c - g_lo * s
dy    = rne_dtype(dy)                          # the same declared-dtype boundary
dot   = tree(sum_k dy[k] * w[k] * x[k])        # the same mid-split tree
coeff = (((dot * r) * r) * r) / 128
dx    = (r * dy) * w - x * coeff               # row-local, no cross-row reduction
dW    = left_fold((dy * x) * r)                # ascending row order, fp32
```

`dx` is row-local (bitwise layout-independent). `dW` per expert is a per-row FP32
contribution followed by an **ascending-row left fold** — on CUDA the
`_C.reduce_rows_fp32_left_fold` kernel, on ROCm the Triton `fold_rows` kernel with
the same ascending order. The rows of the unused expert are multiplied by their
mask first, so an idle expert's gradient is exactly zero.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `q` | `[N, Hq, 128]` | fp32 / bf16 / fp16 | contiguous, unit inner stride |
| `k` | `[N, Hk, 128]` | same as `q` | contiguous, unit inner stride |
| `q_weight_und`, `q_weight_gen` | `[128]` | same as `q` | contiguous; the understanding / generator expert |
| `k_weight_und`, `k_weight_gen` | `[128]` | same as `q` | contiguous |
| `gen_mask` | `[N]` | `torch.bool` | `True` selects the generator expert for that token row |
| `positions` | `[N]` | `torch.int64` | absolute rotary position per token row |
| `cos`, `sin` | `[P, 64]` | `torch.float32` | `P > max(positions)`, contiguous |
| `q_eps`, `k_eps` | scalar | `float` | RMSNorm eps (Qwen3/BAGEL default `1e-6`) |
| `q_out`, `k_out` | as `q`, `k` | as `q`, `k` | newly allocated |

`head_dim` is pinned to `128` (half = `64`). A call the kernels cannot serve
without a cast or copy — non-contiguous, other head dim, fp64, mixed dtypes — is
served by the reference instead, never by a silent conversion.

## Dispatch Behavior

| Platform | Order |
| --- | --- |
| CUDA | `MotQkNormRopeCudaOp` → `MotQkNormRopeTritonOp` → `NativeMotQkNormRopeOp` |
| ROCm | `MotQkNormRopeCudaOp` (hipified `.cu`) → `MotQkNormRopeTritonOp` → `NativeMotQkNormRopeOp` |
| CPU / NPU | `NativeMotQkNormRopeOp` |

The CUDA candidate comes first on ROCm as well: the same source compiled by
`hipcc` implements the identical sequence, so putting it first keeps one
arithmetic contract across the two GPU platforms. When the `_C` symbols are
missing, `MotQkNormRopeCudaOp.__init__` raises and the registry moves on to
Triton.

## Accuracy

Two different promises, and the difference is deliberate:

- **Bitwise invariance (the row's own guarantee).** The operator is row-local, so
  a row's bytes cannot depend on the batch size, the row's position in the batch,
  a pack offset, how many text tokens share the call, the launch geometry, or a
  repeated run. Asserted with `torch.equal` (plus a bit-pattern comparison, so
  `0.0` vs `-0.0` also fails) in `TestMotQkNormRopeInvariance`. The per-expert
  `dW` is the one cross-row reduction, so full-vs-chunked `dW` is compared under
  the contract tolerance instead.
- **Cross-backend equality on one device.** The Triton kernel and the hipified
  CUDA kernel implement the same declared sequence and are asserted
  **byte-identical** to each other (`TestMotQkNormRopeTritonParity`). Cross-
  *platform* byte equality (an NVIDIA GPU vs an AMD GPU) is **not** claimed: no
  evidence backs it until both hosts run the row.
- **Against the independent FP32 reference.** The declared tolerances are the
  contract's `reduction` rows (`ws1.json`): fp32 `atol=rtol=1e-4`, bf16
  `atol=0.05 / rtol=0.02`, fp16 `atol=rtol=1e-3`. `forward_fp32` vs
  `forward_fp64` stays within `1e-5`, which is what makes the FP32 tree itself
  auditable on a CPU-only host.

**FP16 versus the FP32 gold.** The `float16` reduction cell is the one row where
the tolerance and the declared sequence disagree, and no implementation can close
that gap. `forward_fp32` performs no cast before the rotation; the row requires
one. Measured on an H100 against `forward_fp32`: forward max `|candidate - gold|`
is `5.6e-3` for `q` and `3.6e-3` for `k` (and `7.8e-1` relative where the rotated
value cancels toward zero), and per-expert `dW` reaches `0.5` at `|dW| ~ 581` —
one FP16 ulp of the accumulated value. The candidate is **bitwise** equal to
`forward` (the declared-dtype reference path) in fp32, bf16 and fp16 alike, and
CUDA and Triton are bitwise equal to each other. `float16` is `optional` in
`ws1.json`, so the row's gate is fp32 and bf16; `check_operator --dtype fp16`
fails the cell, not the arithmetic.

The reduction tree and the multiplication order are the contract, not an
implementation detail: they are byte-equal between the reference, the CUDA kernel
and the Triton kernel, so a documented drift here is a bug, not a tolerance.

## Performance Notes

```bash
python benchmarks/backends/cuda/benchmark_mot_qk_norm_rope.py \
    --dtype bf16 --rows 4096 --n-heads 28 --n-kv-heads 4 --iters 20 --warmup 5
```

The benchmark reports `fused_ms`, `unfused_ms`, `speedup`, `fused_peak_mb` and
`unfused_peak_mb` for the forward and for forward+backward. The baseline is the
composition the model ran before fusion — a masked `F.rms_norm` per expert (two
calls, one per selected weight) followed by `NativeRoPEOp`, whose rotation is a
broadcast multiply over a duplicated FP32 table rebuilt on every call. The
baseline computes the same operator (it agrees with the reference within `1e-6`
per element on CPU), so the reported ratio is fused-vs-eager work for work.

The fused path touches each row once for both the statistic and the rotation and
launches `q` and `k` from one wrapper; `n_heads=28, n_kv_heads=4, head_dim=128`
is the BAGEL-7B-MoT shape and `rows=4096` is one image-token tier.

## Tests

```bash
# CPU: the reference against itself, the RoPE table, per-expert routing
python -m pytest tests/ops/rope/test_mot_qk_norm_rope.py -q -k Reference

# GPU: gtest against the FP32 gold, per dtype, with gradients
python tools/validation/operators/check_operator.py --op mot_qk_norm_rope \
    --candidate cuda --device cuda --dtype bf16 \
    --batch 2 --seq 16 --n-heads 28 --n-kv-heads 4 --head-dim 128 --check-grad
python tools/validation/operators/check_operator.py --op mot_qk_norm_rope \
    --candidate triton --device cuda --dtype bf16 \
    --batch 2 --seq 16 --n-heads 28 --n-kv-heads 4 --head-dim 128 --check-grad

# GPU: bitwise invariance, and CUDA-vs-Triton byte parity on the same device
python -m pytest tests/ops/rope/test_mot_qk_norm_rope.py -q -k Invariance
python -m pytest tests/ops/rope/test_mot_qk_norm_rope.py -q -k TritonParity

# Registry / CI gate
python -m pytest tests/runtime/test_dispatch.py -q
python -m pytest tests/validation/operators/test_ws1_gtest_gpu.py -q
```

## Known Limitations

- **`head_dim = 128` only.** The kernels are compiled for one row width; other
  head dims go to the reference.
- **Contiguous `[N, H, 128]` fast path.** A non-contiguous view, a mixed-dtype
  call or an fp64 call is served by the reference rather than copied — the
  wrapper refuses instead of silently changing the arithmetic.
- **CUDA/ROCm only.** CPU and NPU run the reference; there is no Ascend kernel
  for this row.
- **No bagel workload/profile row.** The BAGEL end-to-end workload belongs to the
  `bagel_arch_fingerprint` row; this page covers the operator, its backends and
  its own invariance suite.
- **Cross-platform byte equality is unclaimed.** CUDA-vs-Triton parity is asserted
  per device; an NVIDIA-vs-AMD comparison needs both hosts to run the row.
