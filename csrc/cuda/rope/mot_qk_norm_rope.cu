// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Routed (two-expert) FP32 per-head Q/K RMSNorm fused with NeoX RoPE.
//
// One block per (token, head) row of 128 elements: a warp mid-split butterfly
// plus a four-warp combine produce the mean, rstd is 1 / sqrt(mean + eps), the
// normalized value is rounded once to the declared dtype, and the rotation runs
// in FP32 with the FP32 cos/sin table. Plain CUDA C only (no PTX, no TMA) so the
// same source is compiled by hipcc for ROCm, which is what makes the CUDA and
// ROCm kernels byte-identical.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#if !defined(USE_ROCM)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// CUDA's headers define __USE_FAST_MATH__ as 0 when IEEE FP32 is in force, so the
// guard has to test the value, not just its presence.
#if defined(__USE_FAST_MATH__) && __USE_FAST_MATH__
#error "mot_qk_norm_rope requires IEEE FP32 (no fast math)"
#endif

namespace {

constexpr int MOT_QK_NORM_HEAD_DIM = 128;
constexpr int MOT_QK_NORM_HALF = 64;

template <typename scalar_t>
__device__ __forceinline__ float to_f32(const scalar_t value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_f32<at::Half>(const at::Half value) {
    return __half2float(*reinterpret_cast<const __half*>(&value));
}

template <>
__device__ __forceinline__ float to_f32<at::BFloat16>(const at::BFloat16 value) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&value));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_f32(const float value) {
    return static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ at::Half from_f32<at::Half>(const float value) {
    const __half converted = __float2half(value);
    return *reinterpret_cast<const at::Half*>(&converted);
}

template <>
__device__ __forceinline__ at::BFloat16 from_f32<at::BFloat16>(const float value) {
    const __nv_bfloat16 converted = __float2bfloat16(value);
    return *reinterpret_cast<const at::BFloat16*>(&converted);
}

// Mid-split butterfly over the 32 lanes of one warp: the lane offsets ascend
// (1, 2, 4, 8, 16), so lane 0 holds T(0..7) = T(0..3) + T(4..7) at the end --
// the contiguous mid-split association, not a reversed-offset tree. Only lane 0
// of each warp carries the complete warp sum.
__device__ __forceinline__ float warp_sum_mid_split(float value) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <typename scalar_t, typename weight_t>
__global__ void mot_qk_norm_rope_fwd_kernel(
    const scalar_t* __restrict__ x,
    const weight_t* __restrict__ w_und,
    const weight_t* __restrict__ w_gen,
    const bool* __restrict__ gen_mask,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    scalar_t* __restrict__ out,
    float* __restrict__ rstd_out,
    int heads,
    float eps
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t base = (static_cast<int64_t>(token) * heads + head) * MOT_QK_NORM_HEAD_DIM;

    const bool gen = gen_mask[token];
    const weight_t* weight = gen ? w_gen : w_und;

    __shared__ float staged[MOT_QK_NORM_HEAD_DIM];
    __shared__ float warp_sums[4];
    __shared__ float rstd_shared;

    const float xv = to_f32<scalar_t>(x[base + tid]);
    staged[tid] = xv;

    float value = warp_sum_mid_split(__fmul_rn(xv, xv));
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = value;
    }
    __syncthreads();

    if (tid == 0) {
        const float sum01 = __fadd_rn(warp_sums[0], warp_sums[1]);
        const float sum23 = __fadd_rn(warp_sums[2], warp_sums[3]);
        const float mean =
            __fdiv_rn(__fadd_rn(sum01, sum23), static_cast<float>(MOT_QK_NORM_HEAD_DIM));
        rstd_shared = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
        if (rstd_out != nullptr) {
            rstd_out[static_cast<int64_t>(token) * heads + head] = rstd_shared;
        }
    }
    __syncthreads();

    if (tid < MOT_QK_NORM_HALF) {
        const float rstd = rstd_shared;
        const float normed_lo =
            __fmul_rn(__fmul_rn(staged[tid], rstd), to_f32<weight_t>(weight[tid]));
        const float normed_hi = __fmul_rn(
            __fmul_rn(staged[tid + MOT_QK_NORM_HALF], rstd),
            to_f32<weight_t>(weight[tid + MOT_QK_NORM_HALF]));
        // The declared dtype boundary: one rounding before the rotation.
        const float y_lo = to_f32<scalar_t>(from_f32<scalar_t>(normed_lo));
        const float y_hi = to_f32<scalar_t>(from_f32<scalar_t>(normed_hi));

        const int64_t position = positions[token];
        const float cosine = cos_tab[position * MOT_QK_NORM_HALF + tid];
        const float sine = sin_tab[position * MOT_QK_NORM_HALF + tid];
        const float out_lo = __fsub_rn(__fmul_rn(y_lo, cosine), __fmul_rn(y_hi, sine));
        const float out_hi = __fadd_rn(__fmul_rn(y_hi, cosine), __fmul_rn(y_lo, sine));

        out[base + tid] = from_f32<scalar_t>(out_lo);
        out[base + tid + MOT_QK_NORM_HALF] = from_f32<scalar_t>(out_hi);
    }
}

template <typename scalar_t, typename weight_t>
__global__ void mot_qk_norm_rope_bwd_kernel(
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ x,
    const weight_t* __restrict__ w_und,
    const weight_t* __restrict__ w_gen,
    const bool* __restrict__ gen_mask,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    const float* __restrict__ rstd,
    scalar_t* __restrict__ dx,
    float* __restrict__ dw_contrib,
    int heads
) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t row = static_cast<int64_t>(token) * heads + head;
    const int64_t base = row * MOT_QK_NORM_HEAD_DIM;

    const bool gen = gen_mask[token];
    const weight_t* weight = gen ? w_gen : w_und;

    // Inverse rotation: out = R(normed) with R = [[c, -s], [s, c]], so
    // dnormed_lo = g_lo * c + g_hi * s and dnormed_hi = g_hi * c - g_lo * s.
    const int pair = tid & (MOT_QK_NORM_HALF - 1);
    const int64_t position = positions[token];
    const float cosine = cos_tab[position * MOT_QK_NORM_HALF + pair];
    const float sine = sin_tab[position * MOT_QK_NORM_HALF + pair];
    const float g_lo = to_f32<scalar_t>(grad[base + pair]);
    const float g_hi = to_f32<scalar_t>(grad[base + pair + MOT_QK_NORM_HALF]);
    const float dy_rot = (tid < MOT_QK_NORM_HALF)
                             ? __fadd_rn(__fmul_rn(g_lo, cosine), __fmul_rn(g_hi, sine))
                             : __fsub_rn(__fmul_rn(g_hi, cosine), __fmul_rn(g_lo, sine));
    // The same declared dtype boundary the forward rounds through.
    const float dy = to_f32<scalar_t>(from_f32<scalar_t>(dy_rot));

    const float xv = to_f32<scalar_t>(x[base + tid]);
    const float wv = to_f32<weight_t>(weight[tid]);
    const float rstd_row = rstd[row];

    __shared__ float warp_sums[4];
    __shared__ float coeff_shared;

    float value = warp_sum_mid_split(__fmul_rn(__fmul_rn(dy, wv), xv));
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = value;
    }
    __syncthreads();

    if (tid == 0) {
        const float sum01 = __fadd_rn(warp_sums[0], warp_sums[1]);
        const float sum23 = __fadd_rn(warp_sums[2], warp_sums[3]);
        const float dot = __fadd_rn(sum01, sum23);
        coeff_shared = __fdiv_rn(
            __fmul_rn(__fmul_rn(__fmul_rn(dot, rstd_row), rstd_row), rstd_row),
            static_cast<float>(MOT_QK_NORM_HEAD_DIM));
    }
    __syncthreads();

    const float coeff = coeff_shared;
    const float dxv = __fsub_rn(
        __fmul_rn(__fmul_rn(rstd_row, dy), wv),
        __fmul_rn(xv, coeff));
    dx[base + tid] = from_f32<scalar_t>(dxv);
    dw_contrib[base + tid] = __fmul_rn(__fmul_rn(dy, xv), rstd_row);
}

void mot_qk_norm_rope_check(const torch::Tensor& value, const char* name, int dims) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.dim() == dims, name, " must be ", dims, "-D");
}

void mot_qk_norm_rope_check_weights(const torch::Tensor& weight, const char* name) {
    mot_qk_norm_rope_check(weight, name, 1);
    TORCH_CHECK(
        weight.size(0) == MOT_QK_NORM_HEAD_DIM,
        name,
        " must have ",
        MOT_QK_NORM_HEAD_DIM,
        " elements");
}

void mot_qk_norm_rope_check_tables(const torch::Tensor& table, const char* name) {
    mot_qk_norm_rope_check(table, name, 2);
    TORCH_CHECK(
        table.size(1) == MOT_QK_NORM_HALF,
        name,
        " must be [P, ",
        MOT_QK_NORM_HALF,
        "]");
    TORCH_CHECK(table.scalar_type() == torch::kFloat32, name, " must be float32");
}

void mot_qk_norm_rope_launch_forward(
    const torch::Tensor& x,
    const torch::Tensor& w_und,
    const torch::Tensor& w_gen,
    const torch::Tensor& gen_mask,
    const torch::Tensor& positions,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    torch::Tensor& out,
    float* rstd_out,
    int heads,
    float eps,
    cudaStream_t stream
) {
    const dim3 grid(static_cast<unsigned>(x.size(0)), static_cast<unsigned>(heads));
    const dim3 block(MOT_QK_NORM_HEAD_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "mot_qk_norm_rope_forward",
        [&] {
            using x_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf,
                at::kBFloat16,
                w_und.scalar_type(),
                "mot_qk_norm_rope_forward_weight",
                [&] {
                    using w_t = scalar_t;
                    mot_qk_norm_rope_fwd_kernel<x_t, w_t><<<grid, block, 0, stream>>>(
                        x.data_ptr<x_t>(),
                        w_und.data_ptr<w_t>(),
                        w_gen.data_ptr<w_t>(),
                        gen_mask.data_ptr<bool>(),
                        positions.data_ptr<int64_t>(),
                        cos.data_ptr<float>(),
                        sin.data_ptr<float>(),
                        out.data_ptr<x_t>(),
                        rstd_out,
                        heads,
                        eps);
                });
        });
}

void mot_qk_norm_rope_launch_backward(
    const torch::Tensor& grad,
    const torch::Tensor& x,
    const torch::Tensor& w_und,
    const torch::Tensor& w_gen,
    const torch::Tensor& gen_mask,
    const torch::Tensor& positions,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& rstd,
    torch::Tensor& dx,
    torch::Tensor& dw_contrib,
    int heads,
    cudaStream_t stream
) {
    const dim3 grid(static_cast<unsigned>(x.size(0)), static_cast<unsigned>(heads));
    const dim3 block(MOT_QK_NORM_HEAD_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "mot_qk_norm_rope_backward",
        [&] {
            using x_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf,
                at::kBFloat16,
                w_und.scalar_type(),
                "mot_qk_norm_rope_backward_weight",
                [&] {
                    using w_t = scalar_t;
                    mot_qk_norm_rope_bwd_kernel<x_t, w_t><<<grid, block, 0, stream>>>(
                        grad.data_ptr<x_t>(),
                        x.data_ptr<x_t>(),
                        w_und.data_ptr<w_t>(),
                        w_gen.data_ptr<w_t>(),
                        gen_mask.data_ptr<bool>(),
                        positions.data_ptr<int64_t>(),
                        cos.data_ptr<float>(),
                        sin.data_ptr<float>(),
                        rstd.data_ptr<float>(),
                        dx.data_ptr<x_t>(),
                        dw_contrib.data_ptr<float>(),
                        heads);
                });
        });
}

}  // namespace

std::vector<torch::Tensor> mot_qk_norm_rope_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_weight_und,
    torch::Tensor q_weight_gen,
    torch::Tensor k_weight_und,
    torch::Tensor k_weight_gen,
    torch::Tensor gen_mask,
    torch::Tensor positions,
    torch::Tensor cos,
    torch::Tensor sin,
    double q_eps,
    double k_eps,
    bool need_rstd
) {
    mot_qk_norm_rope_check(q, "q", 3);
    mot_qk_norm_rope_check(k, "k", 3);
    TORCH_CHECK(
        q.size(-1) == MOT_QK_NORM_HEAD_DIM,
        "q.shape[-1] must be ",
        MOT_QK_NORM_HEAD_DIM,
        ", got ",
        q.size(-1));
    TORCH_CHECK(
        k.size(-1) == MOT_QK_NORM_HEAD_DIM,
        "k.shape[-1] must be ",
        MOT_QK_NORM_HEAD_DIM,
        ", got ",
        k.size(-1));
    TORCH_CHECK(q.size(0) == k.size(0), "q and k must share the row count");

    const int64_t rows = q.size(0);
    // The kernels are compiled for one weight dtype pair, so this is the one
    // dtype gate the entry point enforces instead of casting silently.
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must share a dtype");
    mot_qk_norm_rope_check_weights(q_weight_und, "q_weight_und");
    mot_qk_norm_rope_check_weights(q_weight_gen, "q_weight_gen");
    mot_qk_norm_rope_check_weights(k_weight_und, "k_weight_und");
    mot_qk_norm_rope_check_weights(k_weight_gen, "k_weight_gen");
    TORCH_CHECK(
        q_weight_und.scalar_type() == q.scalar_type() &&
            q_weight_gen.scalar_type() == q.scalar_type() &&
            k_weight_und.scalar_type() == q.scalar_type() &&
            k_weight_gen.scalar_type() == q.scalar_type(),
        "the four norm weights must share q's dtype");

    mot_qk_norm_rope_check(gen_mask, "gen_mask", 1);
    TORCH_CHECK(gen_mask.size(0) == rows, "gen_mask must have one entry per row");
    TORCH_CHECK(gen_mask.scalar_type() == torch::kBool, "gen_mask must be bool");
    mot_qk_norm_rope_check(positions, "positions", 1);
    TORCH_CHECK(positions.size(0) == rows, "positions must have one entry per row");
    TORCH_CHECK(positions.scalar_type() == torch::kLong, "positions must be int64");
    mot_qk_norm_rope_check_tables(cos, "cos");
    mot_qk_norm_rope_check_tables(sin, "sin");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must share a shape");

    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    const auto rstd_options = q.options().dtype(torch::kFloat32);
    torch::Tensor q_rstd;
    torch::Tensor k_rstd;
    if (need_rstd) {
        q_rstd = torch::empty({rows * q.size(1)}, rstd_options);
        k_rstd = torch::empty({rows * k.size(1)}, rstd_options);
    } else {
        q_rstd = torch::empty({0}, rstd_options);
        k_rstd = torch::empty({0}, rstd_options);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mot_qk_norm_rope_launch_forward(
        q,
        q_weight_und,
        q_weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
        q_out,
        need_rstd ? q_rstd.data_ptr<float>() : nullptr,
        static_cast<int>(q.size(1)),
        static_cast<float>(q_eps),
        stream);
    mot_qk_norm_rope_launch_forward(
        k,
        k_weight_und,
        k_weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
        k_out,
        need_rstd ? k_rstd.data_ptr<float>() : nullptr,
        static_cast<int>(k.size(1)),
        static_cast<float>(k_eps),
        stream);

    return {q_out, k_out, q_rstd, k_rstd};
}

std::vector<torch::Tensor> mot_qk_norm_rope_backward(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_weight_und,
    torch::Tensor q_weight_gen,
    torch::Tensor k_weight_und,
    torch::Tensor k_weight_gen,
    torch::Tensor gen_mask,
    torch::Tensor positions,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor q_rstd,
    torch::Tensor k_rstd
) {
    mot_qk_norm_rope_check(grad_q, "grad_q", 3);
    mot_qk_norm_rope_check(grad_k, "grad_k", 3);
    mot_qk_norm_rope_check(q, "q", 3);
    mot_qk_norm_rope_check(k, "k", 3);
    TORCH_CHECK(grad_q.sizes() == q.sizes(), "grad_q must match q");
    TORCH_CHECK(grad_k.sizes() == k.sizes(), "grad_k must match k");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must share a dtype");
    TORCH_CHECK(grad_q.scalar_type() == q.scalar_type(), "grad_q must share q's dtype");
    TORCH_CHECK(grad_k.scalar_type() == k.scalar_type(), "grad_k must share k's dtype");
    mot_qk_norm_rope_check_weights(q_weight_und, "q_weight_und");
    mot_qk_norm_rope_check_weights(q_weight_gen, "q_weight_gen");
    mot_qk_norm_rope_check_weights(k_weight_und, "k_weight_und");
    mot_qk_norm_rope_check_weights(k_weight_gen, "k_weight_gen");
    mot_qk_norm_rope_check(gen_mask, "gen_mask", 1);
    TORCH_CHECK(gen_mask.scalar_type() == torch::kBool, "gen_mask must be bool");
    mot_qk_norm_rope_check(positions, "positions", 1);
    TORCH_CHECK(positions.scalar_type() == torch::kLong, "positions must be int64");
    mot_qk_norm_rope_check_tables(cos, "cos");
    mot_qk_norm_rope_check_tables(sin, "sin");
    mot_qk_norm_rope_check(q_rstd, "q_rstd", 1);
    mot_qk_norm_rope_check(k_rstd, "k_rstd", 1);
    TORCH_CHECK(q_rstd.scalar_type() == torch::kFloat32, "q_rstd must be float32");
    TORCH_CHECK(k_rstd.scalar_type() == torch::kFloat32, "k_rstd must be float32");
    TORCH_CHECK(
        q_rstd.size(0) == q.size(0) * q.size(1),
        "q_rstd must hold one entry per (token, head) row");
    TORCH_CHECK(
        k_rstd.size(0) == k.size(0) * k.size(1),
        "k_rstd must hold one entry per (token, head) row");

    auto dq = torch::empty_like(q);
    auto dk = torch::empty_like(k);
    const auto contrib_options = q.options().dtype(torch::kFloat32);
    auto q_dw_contrib = torch::empty({q.size(0) * q.size(1), MOT_QK_NORM_HEAD_DIM}, contrib_options);
    auto k_dw_contrib = torch::empty({k.size(0) * k.size(1), MOT_QK_NORM_HEAD_DIM}, contrib_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mot_qk_norm_rope_launch_backward(
        grad_q,
        q,
        q_weight_und,
        q_weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
        q_rstd,
        dq,
        q_dw_contrib,
        static_cast<int>(q.size(1)),
        stream);
    mot_qk_norm_rope_launch_backward(
        grad_k,
        k,
        k_weight_und,
        k_weight_gen,
        gen_mask,
        positions,
        cos,
        sin,
        k_rstd,
        dk,
        k_dw_contrib,
        static_cast<int>(k.size(1)),
        stream);

    return {dq, dk, q_dw_contrib, k_dw_contrib};
}
