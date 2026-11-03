# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tests for the BAGEL MoT routed Q/K RMSNorm + NeoX RoPE row.

Two orthogonal properties:

* correctness against the independent FP32/FP64 reference (``forward_fp32`` is
  the ground truth; ``forward_fp64`` re-runs the declared tree in FP64 so the
  FP32 tree can be checked on a CPU-only host), and
* bitwise invariance: the operator is row-local, so a row's bytes cannot depend
  on batch size, batch position, pack offset, how many text tokens share the
  call, or a repeated run. Cross-backend byte equality (CUDA vs Triton on one
  device) is asserted separately, because the two implement the same declared
  sequence.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from rl_engine.reference.rope.mot_qk_norm_rope import (
    HALF,
    HEAD_DIM,
    NativeMotQkNormRopeOp,
    build_rope_table,
)

GRAD_NAMES = (
    "q",
    "k",
    "q_weight_und",
    "q_weight_gen",
    "k_weight_und",
    "k_weight_gen",
)
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
# The contract's reduction tolerances (ws1.json) for op_class="reduction".
TOLERANCES = {
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (0.05, 0.02),
    torch.float16: (1e-3, 1e-3),
}


def _cuda_kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    return _EXT_AVAILABLE and hasattr(_C, "mot_qk_norm_rope_forward")


def _triton_available() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


requires_cuda = pytest.mark.skipif(
    not _cuda_kernel_available(), reason="needs a GPU plus the mot_qk_norm_rope symbols"
)
requires_triton = pytest.mark.skipif(
    not (_cuda_kernel_available() and _triton_available()),
    reason="needs the CUDA kernel and Triton to compare them",
)


def _gen_mask_pattern(pattern: str, rows: int, device: torch.device) -> torch.Tensor:
    if pattern == "alternating":
        return torch.arange(rows, device=device) % 2 == 1
    if pattern == "all_und":
        return torch.zeros(rows, device=device, dtype=torch.bool)
    if pattern == "all_gen":
        return torch.ones(rows, device=device, dtype=torch.bool)
    if pattern == "first_half":
        mask = torch.zeros(rows, device=device, dtype=torch.bool)
        mask[: max(1, rows // 2)] = True
        return mask
    raise ValueError(f"unknown gen pattern {pattern!r}")


def _make_inputs(
    dtype: torch.dtype = torch.float32,
    *,
    rows: int = 8,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    gen_pattern: str = "alternating",
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, object]:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = [
        torch.randn((HEAD_DIM,), generator=generator, device=device, dtype=dtype) for _ in range(4)
    ]
    cos, sin = build_rope_table(rows, HEAD_DIM, device=device)
    return {
        "q": torch.randn(
            (rows, n_heads, HEAD_DIM), generator=generator, device=device, dtype=dtype
        ),
        "k": torch.randn(
            (rows, n_kv_heads, HEAD_DIM), generator=generator, device=device, dtype=dtype
        ),
        "q_weight_und": weights[0],
        "q_weight_gen": weights[1],
        "k_weight_und": weights[2],
        "k_weight_gen": weights[3],
        "gen_mask": _gen_mask_pattern(gen_pattern, rows, device),
        "positions": torch.arange(rows, device=device, dtype=torch.long),
        "cos": cos,
        "sin": sin,
        "q_eps": 1e-6,
        "k_eps": 1e-6,
    }


def _manual_reference(
    x: torch.Tensor,
    weight_und: torch.Tensor,
    weight_gen: torch.Tensor,
    gen_mask: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """A second FP32 implementation written from the contract, not `_apply`."""

    x32 = x.float()
    weight = torch.where(gen_mask[:, None], weight_gen.float(), weight_und.float())
    squared = x32.square()
    for _ in range(7):
        squared = squared.reshape(*squared.shape[:-1], -1, 2).sum(-1)
    rstd = 1.0 / torch.sqrt(squared / float(HEAD_DIM) + float(eps))
    normed = (x32 * rstd) * weight[:, None, :]
    low, high = normed[..., :HALF], normed[..., HALF:]
    cosine = cos[positions].unsqueeze(1)
    sine = sin[positions].unsqueeze(1)
    return torch.cat([low * cosine - high * sine, high * cosine + low * sine], dim=-1)


def _cuda_op():
    from rl_engine.backends.cuda.rope.mot_qk_norm_rope import MotQkNormRopeCudaOp

    return MotQkNormRopeCudaOp()


def _triton_op():
    from rl_engine.backends.shared.triton.rope.mot_qk_norm_rope import MotQkNormRopeTritonOp

    return MotQkNormRopeTritonOp()


def _unsqueeze_inputs(inputs: dict[str, object]) -> dict[str, object]:
    args = dict(inputs)
    for name in GRAD_NAMES:
        args[name] = inputs[name].detach().clone().requires_grad_(True)
    return args


def _forward_backward(op, inputs: dict[str, object], *, seed: int = 7, grad_mode: str = "random"):
    """Forward plus the six grads under one upstream gradient.

    ``grad_mode="ones"`` is what the invariance tests use: a row's ``dq``/``dk``
    must be comparable across shapes, so the upstream gradient may not depend on
    the row count or on the row index.
    """

    args = _unsqueeze_inputs(inputs)
    q_out, k_out = op.forward(**args)
    if grad_mode == "ones":
        grad_q = torch.ones_like(q_out)
        grad_k = torch.ones_like(k_out)
    elif grad_mode == "random":
        generator = torch.Generator(device=q_out.device).manual_seed(seed)
        grad_q = torch.randn(
            q_out.shape, generator=generator, device=q_out.device, dtype=q_out.dtype
        )
        grad_k = torch.randn(
            k_out.shape, generator=generator, device=k_out.device, dtype=k_out.dtype
        )
    else:
        raise ValueError(f"unknown grad_mode {grad_mode!r}")
    grads = torch.autograd.grad(
        (q_out, k_out),
        [args[name] for name in GRAD_NAMES],
        grad_outputs=(grad_q, grad_k),
    )
    return (q_out, k_out), tuple(grads)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """`torch.equal` plus a bit-pattern comparison, so 0.0 vs -0.0 also fails."""

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), (
        f"values differ (max abs " f"{(actual.float() - expected.float()).abs().max().item():.8e})"
    )
    int_dtype = torch.int32 if actual.element_size() == 4 else torch.int16
    assert torch.equal(actual.view(int_dtype), expected.view(int_dtype)), "bit patterns differ"


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    atol, rtol = TOLERANCES[dtype]
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), (
        f"max abs {(actual.float() - expected.float()).abs().max().item():.8e} "
        f"exceeds atol={atol} rtol={rtol}"
    )


class TestMotQkNormRopeReference:
    """CPU-only: the reference against itself, the RoPE table, and routing."""

    def _inputs(self, *, dtype: torch.dtype = torch.float32, rows: int = 8, seed: int = 0):
        return _make_inputs(dtype, rows=rows, n_heads=8, n_kv_heads=4, seed=seed)

    def test_forward_fp32_agrees_with_fp64(self):
        inputs = self._inputs()
        op = NativeMotQkNormRopeOp()
        fp32 = op.forward_fp32(**inputs)
        fp64 = op.forward_fp64(**inputs)
        for value32, value64 in zip(fp32, fp64, strict=True):
            assert value32.dtype is torch.float32
            assert value64.dtype is torch.float64
            max_abs = (value32.double() - value64).abs().max().item()
            assert max_abs <= 1e-5, f"FP32 tree drifts {max_abs:.3e} from its FP64 twin"

    def test_rope_table_matches_native_rope(self):
        from rl_engine.reference.rope.rope import NativeRoPEOp

        positions = torch.arange(128)
        cos, sin = build_rope_table(128, HEAD_DIM, device="cpu")
        ref_cos, ref_sin = NativeRoPEOp._compute_cos_sin(
            torch.zeros(1, 1, HEAD_DIM), positions, theta=1e6
        )
        # The native table duplicates each half; the row stores one half.
        assert torch.equal(cos, ref_cos.squeeze(0).squeeze(0)[:, :HALF])
        assert torch.equal(sin, ref_sin.squeeze(0).squeeze(0)[:, :HALF])
        assert cos.shape == (128, HALF) and cos.dtype is torch.float32

    @pytest.mark.parametrize("pattern", ["all_und", "all_gen"])
    def test_routing_selects_the_per_expert_weight(self, pattern):
        inputs = self._inputs()
        inputs["gen_mask"] = _gen_mask_pattern(pattern, inputs["q"].shape[0], torch.device("cpu"))
        op = NativeMotQkNormRopeOp()
        q_out, k_out = op.forward_fp32(**inputs)
        expected_q = _manual_reference(
            inputs["q"],
            inputs["q_weight_und"],
            inputs["q_weight_gen"],
            inputs["gen_mask"],
            inputs["positions"],
            inputs["cos"],
            inputs["sin"],
            inputs["q_eps"],
        )
        expected_k = _manual_reference(
            inputs["k"],
            inputs["k_weight_und"],
            inputs["k_weight_gen"],
            inputs["gen_mask"],
            inputs["positions"],
            inputs["cos"],
            inputs["sin"],
            inputs["k_eps"],
        )
        _assert_bitwise_equal(q_out, expected_q)
        _assert_bitwise_equal(k_out, expected_k)

    def test_mixed_modality_rows_are_row_local(self):
        rows = 8
        base = _make_inputs(torch.float32, rows=rows, n_heads=8, n_kv_heads=4, seed=5)
        base["gen_mask"] = _gen_mask_pattern("alternating", rows, torch.device("cpu"))
        op = NativeMotQkNormRopeOp()
        q_small, k_small = op.forward_fp32(**base)

        # Same first four rows, twelve extra text rows appended.
        extended = _make_inputs(torch.float32, rows=rows + 12, n_heads=8, n_kv_heads=4, seed=5)
        extended["gen_mask"] = _gen_mask_pattern("alternating", rows + 12, torch.device("cpu"))
        extended["q"][:4] = base["q"][:4]
        extended["k"][:4] = base["k"][:4]
        extended["gen_mask"][:4] = base["gen_mask"][:4]
        q_large, k_large = op.forward_fp32(**extended)

        _assert_bitwise_equal(q_large[:4], q_small[:4])
        _assert_bitwise_equal(k_large[:4], k_small[:4])

    def test_declared_path_is_a_no_op_for_fp32(self):
        inputs = self._inputs()
        op = NativeMotQkNormRopeOp()
        declared = op.forward(**inputs)
        fp32 = op.forward_fp32(**inputs)
        for value, expected in zip(declared, fp32, strict=True):
            _assert_bitwise_equal(value, expected)

    def test_validation_reports_the_failing_field(self):
        inputs = self._inputs()
        op = NativeMotQkNormRopeOp()
        with pytest.raises(ValueError, match="q.shape\\[-1\\]"):
            op.forward_fp32(**{**inputs, "q": inputs["q"][..., :64]})
        with pytest.raises(ValueError, match="gen_mask"):
            op.forward_fp32(**{**inputs, "gen_mask": inputs["gen_mask"].long()})
        with pytest.raises(ValueError, match="cos/sin must be float32"):
            op.forward_fp32(**{**inputs, "cos": inputs["cos"].double()})


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
class TestMotQkNormRopeCudaCorrectness:
    """The CUDA kernel against the FP32 reference, per dtype."""

    def test_forward_matches_gold(self, dtype):
        inputs = _make_inputs(dtype, rows=8, n_heads=8, n_kv_heads=2, device="cuda")
        op = _cuda_op()
        q_out, k_out = op.forward(**inputs)
        gold_q, gold_k = NativeMotQkNormRopeOp().forward_fp32(**inputs)
        assert q_out.dtype is dtype and k_out.dtype is dtype
        _assert_close(q_out, gold_q, dtype)
        _assert_close(k_out, gold_k, dtype)

    def test_forward_fp32_matches_gold(self, dtype):
        inputs = _make_inputs(dtype, rows=8, n_heads=4, n_kv_heads=2, device="cuda")
        op = _cuda_op()
        out = op.forward_fp32(**inputs)
        gold = NativeMotQkNormRopeOp().forward_fp32(**inputs)
        for value, expected in zip(out, gold, strict=True):
            _assert_close(value, expected, dtype)

    def test_backward_matches_gold(self, dtype):
        inputs = _make_inputs(
            dtype, rows=8, n_heads=8, n_kv_heads=2, gen_pattern="alternating", device="cuda"
        )
        (q_out, k_out), grads = _forward_backward(_cuda_op(), inputs)
        (gold_q, gold_k), gold_grads = _forward_backward(NativeMotQkNormRopeOp(), inputs)
        _assert_close(q_out, gold_q, dtype)
        _assert_close(k_out, gold_k, dtype)
        for name, value, expected in zip(GRAD_NAMES, grads, gold_grads, strict=True):
            assert value is not None and value.shape == expected.shape, name
            _assert_close(value, expected, dtype)

    def test_head_shapes(self, dtype):
        for n_heads, n_kv_heads in ((28, 4), (8, 2), (4, 1)):
            inputs = _make_inputs(
                dtype, rows=6, n_heads=n_heads, n_kv_heads=n_kv_heads, device="cuda"
            )
            q_out, k_out = _cuda_op().forward(**inputs)
            gold_q, gold_k = NativeMotQkNormRopeOp().forward_fp32(**inputs)
            assert q_out.shape == (6, n_heads, HEAD_DIM)
            assert k_out.shape == (6, n_kv_heads, HEAD_DIM)
            _assert_close(q_out, gold_q, dtype)
            _assert_close(k_out, gold_k, dtype)

    @pytest.mark.parametrize("pattern", ["alternating", "all_und", "all_gen"])
    def test_expert_patterns(self, dtype, pattern):
        inputs = _make_inputs(
            dtype, rows=8, n_heads=4, n_kv_heads=2, gen_pattern=pattern, device="cuda"
        )
        (q_out, k_out), grads = _forward_backward(_cuda_op(), inputs)
        (gold_q, gold_k), gold_grads = _forward_backward(NativeMotQkNormRopeOp(), inputs)
        _assert_close(q_out, gold_q, dtype)
        _assert_close(k_out, gold_k, dtype)
        for value, expected in zip(grads[2:], gold_grads[2:], strict=True):
            _assert_close(value, expected, dtype)
        # An expert that serves no row has the exact zero gradient.
        idle = {
            "all_und": (3, 5),
            "all_gen": (2, 4),
        }.get(pattern)
        if idle is not None:
            for index in idle:
                assert torch.equal(grads[index], torch.zeros_like(grads[index])), GRAD_NAMES[index]


@requires_cuda
class TestMotQkNormRopeInvariance:
    """Bitwise row locality: the operator's own guarantee."""

    def test_batch_invariance(self):
        rows, n_heads, n_kv_heads = 6, 4, 2
        inputs = _make_inputs(
            torch.float32, rows=rows, n_heads=n_heads, n_kv_heads=n_kv_heads, device="cuda"
        )
        (q_full, k_full), grads_full = _forward_backward(_cuda_op(), inputs, grad_mode="ones")
        for row in range(rows):
            single = {
                **inputs,
                "q": inputs["q"][row : row + 1],
                "k": inputs["k"][row : row + 1],
                "gen_mask": inputs["gen_mask"][row : row + 1],
                "positions": inputs["positions"][row : row + 1],
            }
            (q_row, k_row), grads_row = _forward_backward(_cuda_op(), single, grad_mode="ones")
            _assert_bitwise_equal(q_row, q_full[row : row + 1])
            _assert_bitwise_equal(k_row, k_full[row : row + 1])
            for full, one in zip(grads_full[:2], grads_row[:2], strict=True):
                _assert_bitwise_equal(one, full[row : row + 1])

    def test_permutation_invariance(self):
        rows, n_heads, n_kv_heads = 8, 4, 2
        inputs = _make_inputs(
            torch.float32, rows=rows, n_heads=n_heads, n_kv_heads=n_kv_heads, device="cuda"
        )
        (q_ref, k_ref), grads_ref = _forward_backward(_cuda_op(), inputs, grad_mode="ones")
        permutation = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4], device="cuda")
        permuted = {
            **inputs,
            "q": inputs["q"][permutation],
            "k": inputs["k"][permutation],
            "gen_mask": inputs["gen_mask"][permutation],
            "positions": inputs["positions"][permutation],
        }
        (q_perm, k_perm), grads_perm = _forward_backward(_cuda_op(), permuted, grad_mode="ones")
        inverse = torch.argsort(permutation)
        _assert_bitwise_equal(q_perm[inverse], q_ref)
        _assert_bitwise_equal(k_perm[inverse], k_ref)
        for ref, perm in zip(grads_ref[:2], grads_perm[:2], strict=True):
            _assert_bitwise_equal(perm[inverse], ref)

    def test_pack_offset_is_invisible(self):
        rows, n_heads, n_kv_heads = 6, 4, 2
        inputs = _make_inputs(
            torch.float32, rows=rows + 4, n_heads=n_heads, n_kv_heads=n_kv_heads, device="cuda"
        )
        packed = {
            **inputs,
            "q": inputs["q"][4:],
            "k": inputs["k"][4:],
            "gen_mask": inputs["gen_mask"][4:],
            "positions": inputs["positions"][4:],
        }
        (q_pack, k_pack), grads_pack = _forward_backward(_cuda_op(), packed, grad_mode="ones")
        (q_full, k_full), grads_full = _forward_backward(_cuda_op(), inputs, grad_mode="ones")
        _assert_bitwise_equal(q_pack, q_full[4:])
        _assert_bitwise_equal(k_pack, k_full[4:])
        for one, full in zip(grads_pack[:2], grads_full[:2], strict=True):
            _assert_bitwise_equal(one, full[4:])

    def test_modality_mix_keeps_the_latent_rows(self):
        rows, n_heads, n_kv_heads = 6, 4, 2
        inputs = _make_inputs(
            torch.float32, rows=rows, n_heads=n_heads, n_kv_heads=n_kv_heads, device="cuda"
        )
        inputs["gen_mask"] = _gen_mask_pattern("first_half", rows, torch.device("cuda"))
        q_small, k_small = _cuda_op().forward(**inputs)
        extended = {
            **inputs,
            "q": torch.cat([inputs["q"], inputs["q"][:2]], dim=0),
            "k": torch.cat([inputs["k"], inputs["k"][:2]], dim=0),
            "gen_mask": torch.cat(
                [inputs["gen_mask"], torch.zeros(2, device="cuda", dtype=torch.bool)]
            ),
            "positions": torch.arange(rows + 2, device="cuda", dtype=torch.long),
        }
        q_large, k_large = _cuda_op().forward(**extended)
        _assert_bitwise_equal(q_large[:rows], q_small)
        _assert_bitwise_equal(k_large[:rows], k_small)

    def test_repeated_run_is_identical(self):
        inputs = _make_inputs(torch.float32, rows=8, n_heads=4, n_kv_heads=2, device="cuda")
        first = _cuda_op().forward(**inputs)
        second = _cuda_op().forward(**inputs)
        for value, expected in zip(first, second, strict=True):
            _assert_bitwise_equal(value, expected)

    def test_dweight_full_versus_chunked(self):
        rows, n_heads, n_kv_heads = 8, 4, 2
        inputs = _make_inputs(
            torch.float32, rows=rows, n_heads=n_heads, n_kv_heads=n_kv_heads, device="cuda"
        )
        _outputs, grads = _forward_backward(_cuda_op(), inputs, grad_mode="ones")
        half = rows // 2
        chunked = []
        for start in (0, half):
            chunk = {
                **inputs,
                "q": inputs["q"][start : start + half],
                "k": inputs["k"][start : start + half],
                "gen_mask": inputs["gen_mask"][start : start + half],
                "positions": inputs["positions"][start : start + half],
            }
            chunked.append(_forward_backward(_cuda_op(), chunk, grad_mode="ones")[1])
        for index in range(2, len(GRAD_NAMES)):
            summed = (chunked[0][index].float() + chunked[1][index].float()).to(grads[index].dtype)
            # Parameter grads are cross-row folds: the association differs, so the
            # comparison is the contract tolerance, not bit equality.
            _assert_close(summed, grads[index], torch.float32)


@requires_triton
class TestMotQkNormRopeTritonParity:
    """The Triton and hipified-CUDA kernels implement the same declared sequence."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_forward_and_backward_are_bit_identical(self, dtype):
        runs = ((28, 4), (8, 2))
        for n_heads, n_kv_heads in runs:
            inputs = _make_inputs(
                dtype,
                rows=8,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                gen_pattern="alternating",
                device="cuda",
            )
            (q_cuda, k_cuda), grads_cuda = _forward_backward(_cuda_op(), inputs)
            (q_triton, k_triton), grads_triton = _forward_backward(_triton_op(), inputs)
            _assert_bitwise_equal(q_triton, q_cuda)
            _assert_bitwise_equal(k_triton, k_cuda)
            for value, expected in zip(grads_triton, grads_cuda, strict=True):
                _assert_bitwise_equal(value, expected)

    def test_launch_geometry_is_invisible(self):
        from rl_engine.backends.shared.triton.rope.mot_qk_norm_rope import (
            _launch_backward,
            _launch_forward,
        )

        inputs = _make_inputs(torch.float32, rows=8, n_heads=4, n_kv_heads=2, device="cuda")
        outputs = []
        for num_warps in (1, 2, 4):
            q_out, q_rstd = _launch_forward(
                inputs["q"],
                inputs["q_weight_und"],
                inputs["q_weight_gen"],
                inputs["gen_mask"],
                inputs["positions"],
                inputs["cos"],
                inputs["sin"],
                inputs["q_eps"],
                store_rstd=True,
                num_warps=num_warps,
            )
            k_out, k_rstd = _launch_forward(
                inputs["k"],
                inputs["k_weight_und"],
                inputs["k_weight_gen"],
                inputs["gen_mask"],
                inputs["positions"],
                inputs["cos"],
                inputs["sin"],
                inputs["k_eps"],
                store_rstd=True,
                num_warps=num_warps,
            )
            grad_q = torch.ones_like(q_out)
            grad_k = torch.ones_like(k_out)
            dq, _ = _launch_backward(
                grad_q,
                inputs["q"],
                inputs["q_weight_und"],
                inputs["q_weight_gen"],
                inputs["gen_mask"],
                inputs["positions"],
                inputs["cos"],
                inputs["sin"],
                q_rstd,
                num_warps=num_warps,
            )
            dk, _ = _launch_backward(
                grad_k,
                inputs["k"],
                inputs["k_weight_und"],
                inputs["k_weight_gen"],
                inputs["gen_mask"],
                inputs["positions"],
                inputs["cos"],
                inputs["sin"],
                k_rstd,
                num_warps=num_warps,
            )
            outputs.append((q_out, k_out, dq, dk))
        for other in outputs[1:]:
            for value, expected in zip(other, outputs[0], strict=True):
                _assert_bitwise_equal(value, expected)

    def test_triton_matches_gold_on_the_same_device(self):
        inputs = _make_inputs(torch.bfloat16, rows=8, n_heads=4, n_kv_heads=2, device="cuda")
        q_out, k_out = _triton_op().forward(**inputs)
        gold_q, gold_k = NativeMotQkNormRopeOp().forward_fp32(**inputs)
        _assert_close(q_out, gold_q, torch.bfloat16)
        _assert_close(k_out, gold_k, torch.bfloat16)


class TestMotQkNormRopeRegistry:
    def test_cpu_dispatch_uses_the_reference(self):
        from rl_engine.runtime.registry import kernel_registry

        op = kernel_registry.get_op("mot_qk_norm_rope", device="cpu")
        assert type(op).__name__ == "NativeMotQkNormRopeOp"

    @requires_cuda
    def test_cuda_dispatch_uses_the_kernel(self):
        from rl_engine.runtime.registry import kernel_registry

        op = kernel_registry.get_op("mot_qk_norm_rope", device="cuda")
        assert type(op).__name__ == "MotQkNormRopeCudaOp"
        inputs = _make_inputs(torch.bfloat16, rows=4, n_heads=4, n_kv_heads=2, device="cuda")
        q_out, k_out = op(**inputs)
        gold_q, gold_k = NativeMotQkNormRopeOp().forward_fp32(**inputs)
        _assert_close(q_out, gold_q, torch.bfloat16)
        _assert_close(k_out, gold_k, torch.bfloat16)
