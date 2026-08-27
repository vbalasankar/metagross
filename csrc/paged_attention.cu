#include "metagross_ops.h"
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <tuple>
#include <limits>
#include <pybind11/stl.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxHeadDim = 64;


constexpr int kMaxPageSize = 64;
constexpr int kMaxSeqLen = 4096;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int b = kWarpSize / 2; b > 0; b >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, b);
    }
    return val;
}

__global__ void paged_attention_committed_kernel(
    const float* __restrict__ aj,
    const int8_t* __restrict__ m,
    const int8_t* __restrict__ ax,
    const float* __restrict__ ac,
    const float* __restrict__ ad,
    const int* __restrict__ c,
    int r,
    int w,
    int u,
    int k,
    int ae,
    float l,
    float* __restrict__ aa,
    float* __restrict__ z,
    float* __restrict__ y
) {
    const int ai = blockIdx.x;


    const int o = w / u;
    const int n = ai / o;
    const int tid = threadIdx.x;

    __shared__ float av[kMaxPageSize * kMaxHeadDim];
    __shared__ float au[kMaxSeqLen];
    __shared__ float ar[kMaxHeadDim];

    for (int d = tid; d < k; d += kWarpSize) {
        ar[d] = aj[ai * k + d];
    }
    __syncthreads();


    const int ag = u * k;
    float ak = -INFINITY;


    for (int pg = 0; pg < r; pg++) {
        int ah = c[pg];
        float ao = ac[ah];
        const int8_t* ab = m + (size_t)ah * ae * ag;

        for (int i = tid; i < ae * k; i += kWarpSize) {
            int aw = i / k;
            int d = i % k;
            int8_t raw = ab[aw * ag + n * k + d];
            av[aw * k + d] = static_cast<float>(raw) * ao;
        }
        __syncthreads();

        for (int aw = 0; aw < ae; aw++) {
            float af = 0.0f;
            for (int d = tid; d < k; d += kWarpSize) {
                af += ar[d] * av[aw * k + d];
            }
            float aq = warp_reduce_sum(af) * l;
            aq = __shfl_sync(0xffffffff, aq, 0);

            int f = pg * ae + aw;
            if (tid == 0) {
                au[f] = aq;
            }
            ak = fmaxf(ak, aq);
        }
        __syncthreads();
    }


    float al = 0.0f;
    float am[kMaxHeadDim / kWarpSize];
    for (int j = 0; j < kMaxHeadDim / kWarpSize; j++) {
        am[j] = 0.0f;
    }

    for (int pg = 0; pg < r; pg++) {
        int ah = c[pg];
        float ap = ad[ah];
        const int8_t* ab = ax + (size_t)ah * ae * ag;

        for (int i = tid; i < ae * k; i += kWarpSize) {
            int aw = i / k;
            int d = i % k;
            int8_t raw = ab[aw * ag + n * k + d];
            av[aw * k + d] = static_cast<float>(raw) * ap;
        }
        __syncthreads();

        for (int aw = 0; aw < ae; aw++) {
            int f = pg * ae + aw;
            float ay = expf(au[f] - ak);
            if (tid == 0) {
                al += ay;
            }
            int j = 0;
            for (int d = tid; d < k; d += kWarpSize, j++) {
                am[j] += ay * av[aw * k + d];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        z[ai] = al;
        y[ai] = ak;
    }
    int j = 0;
    for (int d = tid; d < k; d += kWarpSize, j++) {
        aa[ai * k + d] = am[j];
    }
}

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> paged_attention_committed_cuda(
    torch::Tensor bt,
    torch::Tensor bg,
    torch::Tensor bv,
    torch::Tensor bo,
    torch::Tensor bq,
    torch::Tensor az,
    double bf
) {
    TORCH_CHECK(bt.is_cuda() && bg.is_cuda() && bv.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(bo.is_cuda() && bq.is_cuda() && az.is_cuda(),
                "page_scales_k/v and block_table must be CUDA tensors");
    TORCH_CHECK(bt.scalar_type() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(bg.scalar_type() == torch::kInt8 && bv.scalar_type() == torch::kInt8,
                "k_storage/v_storage must be int8");
    TORCH_CHECK(bo.scalar_type() == torch::kFloat32 && bq.scalar_type() == torch::kFloat32,
                "page_scales_k/v must be float32");
    TORCH_CHECK(az.scalar_type() == torch::kInt32, "block_table must be int32");


    TORCH_CHECK(bt.dim() == 2, "query must be 2-D [num_q_heads, head_dim], got ", bt.dim(), "-D");
    TORCH_CHECK(bg.dim() == 4,
                "k_storage must be 4-D [max_pages, page_size, num_kv_heads, head_dim], got ", bg.dim(), "-D");
    TORCH_CHECK(bv.dim() == 4,
                "v_storage must be 4-D [max_pages, page_size, num_kv_heads, head_dim], got ", bv.dim(), "-D");
    TORCH_CHECK(az.dim() == 1, "block_table must be 1-D [num_committed_pages], got ", az.dim(), "-D");
    TORCH_CHECK(bo.dim() == 1 && bq.dim() == 1, "page_scales_k/v must be 1-D [max_pages]");


    TORCH_CHECK(bg.is_contiguous(), "k_storage must be contiguous");
    TORCH_CHECK(bv.is_contiguous(), "v_storage must be contiguous");


    auto bu = bt.contiguous();
    auto ba = az.contiguous();
    auto bp = bo.contiguous();
    auto br = bq.contiguous();


    int64_t bh = bg.size(0);
    TORCH_CHECK(bv.size(0) == bh,
                "k_storage and v_storage must have the same number of physical pages, got ",
                bh, " and ", bv.size(0));
    TORCH_CHECK(bp.size(0) == bh && br.size(0) == bh,
                "page_scales_k/v must have length max_pages=", bh,
                ", got ", bp.size(0), " and ", br.size(0));

    int bs = bg.size(1);
    TORCH_CHECK(bs > 0, "page_size must be positive, got ", bs);
    TORCH_CHECK(bv.size(1) == bs,
                "k_storage and v_storage must have the same page_size, got ", bs,
                " and ", bv.size(1));
    TORCH_CHECK(bs <= kMaxPageSize,
                "page_size (", bs, ") exceeds this kernel's maximum supported page_size (",
                kMaxPageSize, ") -- see file header, 'Simplifying assumptions'");

    TORCH_CHECK(bu.size(1) == kMaxHeadDim,
                "this kernel hardcodes head_dim=", kMaxHeadDim, ", got ", bu.size(1),
                " -- see file header, 'Simplifying assumptions'");
    TORCH_CHECK(bg.size(3) == kMaxHeadDim && bv.size(3) == kMaxHeadDim,
                "k_storage/v_storage head_dim must also be ", kMaxHeadDim);
    TORCH_CHECK(bg.size(2) == bv.size(2), "k_storage and v_storage must have the same num_kv_heads");

    int bk = bu.size(0);
    int bj = bg.size(2);
    int bd = bu.size(1);
    int bi = ba.size(0);

    TORCH_CHECK(
        bj > 0 && bk % bj == 0,
        "num_q_heads (", bk, ") must be an exact multiple of num_kv_heads (", bj,
        ") for grouped-query attention -- got a non-integer group size"
    );
    TORCH_CHECK(
        (int64_t)bi * bs <= kMaxSeqLen,
        "sequence too long for this kernel's static scores buffer: ", (int64_t)bi * bs,
        " committed tokens > kMaxSeqLen=", kMaxSeqLen, " -- see file header, 'Simplifying assumptions'"
    );


    TORCH_CHECK(std::isfinite(bf), "attention scale must be finite, got ", bf);
    TORCH_CHECK(bf > 0.0, "attention scale must be positive, got ", bf);


    if (bi > 0) {
        auto bb = ba.to(torch::kCPU);
        auto bc = bb.accessor<int32_t, 1>();
        for (int64_t i = 0; i < bi; i++) {
            int32_t p = bc[i];
            TORCH_CHECK(
                p >= 0 && p < bh,
                "block_table[", i, "] = ", p, " is out of range for max_pages=", bh
            );
        }
    }

    auto bn = torch::zeros({bk, bd}, bu.options());
    auto bm = torch::zeros({bk}, bu.options());
    auto bl = torch::full({bk}, -std::numeric_limits<float>::infinity(), bu.options());

    if (bi > 0) {
        paged_attention_committed_kernel<<<bk, kWarpSize>>>(
            bu.data_ptr<float>(),
            bg.data_ptr<int8_t>(), bv.data_ptr<int8_t>(),
            bp.data_ptr<float>(), br.data_ptr<float>(),
            ba.data_ptr<int>(),
            bi, bk, bj, bd, bs,
            static_cast<float>(bf),
            bn.data_ptr<float>(), bm.data_ptr<float>(), bl.data_ptr<float>()
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "paged_attention_committed_kernel launch failed: ", cudaGetErrorString(err));
    }


    return {bn, bm, bl};
}
