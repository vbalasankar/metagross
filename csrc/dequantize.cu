#include "metagross_ops.h"
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kDequantizeThreads = 256;

__global__ void dequantize_kernel(
    const int8_t* e,
    float* c,
    int j,
    int b,
    int f,
    float g
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = b * f;
    if (idx >= k) return;

    int i = idx / f;
    int rem = idx % f;

    int d = (j + i) * f + rem;
    c[idx] = static_cast<float>(e[d]) * g;
}

}

torch::Tensor dequantize_page_cuda(
    torch::Tensor q, int64_t u, int64_t o, double t
) {
    TORCH_CHECK(q.is_cuda(), "page_storage must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kInt8, "page_storage must be int8");
    TORCH_CHECK(q.dim() == 3, "page_storage must be [page_size, num_heads, head_dim], got ",
                q.dim(), "-D");


    TORCH_CHECK(q.is_contiguous(), "page_storage must be contiguous");
    TORCH_CHECK(u >= 0, "token_offset_in_page must be >= 0, got ", u);
    TORCH_CHECK(o >= 0, "num_tokens must be >= 0, got ", o);
    TORCH_CHECK(std::isfinite(t) && t > 0.0, "scale must be a positive finite number, got ", t);
    TORCH_CHECK(
        u + o <= q.size(0),
        "dequantize_page_cuda: read would overflow the page (offset=", u,
        ", num_tokens=", o, ", page_size=", q.size(0), ")"
    );

    int n = q.size(1);
    int m = q.size(2);
    int r = n * m;
    int v = static_cast<int>(o) * r;

    auto p = torch::empty({o, n, m}, q.options().dtype(torch::kFloat32));


    if (v == 0) {
        return p;
    }

    const int l = (v + kDequantizeThreads - 1) / kDequantizeThreads;
    dequantize_kernel<<<l, kDequantizeThreads>>>(
        q.data_ptr<int8_t>(), p.data_ptr<float>(),
        static_cast<int>(u), static_cast<int>(o), r,
        static_cast<float>(t)
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dequantize_kernel launch failed: ", cudaGetErrorString(err));

    return p;
}
