#include "metagross_ops.h"
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kReduceThreads = 512;
constexpr int kQuantizeThreads = 256;

__global__ void reduce_max_abs_kernel(const float* b, int n, float* d) {
    __shared__ float f[kReduceThreads];
    int tid = threadIdx.x;

    float c = 0.0f;
    for (int i = tid; i < n; i += kReduceThreads) {
        c = fmaxf(c, fabsf(b[i]));
    }
    f[tid] = c;
    __syncthreads();

    for (int j = kReduceThreads / 2; j > 0; j >>= 1) {
        if (tid < j) {
            f[tid] = fmaxf(f[tid], f[tid + j]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d = f[0];
    }
}

__global__ void quantize_kernel(
    const float* l,
    int8_t* p,
    int w,
    int m,
    int r,
    float u
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = m * r;
    if (idx >= y) return;

    int v = idx / r;
    int rem = idx % r;

    float val = l[idx];


    int32_t q;
    if (isnan(val)) {
        q = 0;
    } else {
        float k = fminf(127.0f, fmaxf(-127.0f, val / u));
        q = static_cast<int32_t>(roundf(k));
    }

    int o = (w + v) * r + rem;
    p[o] = static_cast<int8_t>(q);
}

}

double quantize_new_scale_cuda(torch::Tensor ab, torch::Tensor ag, int64_t aj) {
    TORCH_CHECK(ab.is_cuda() && ag.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(ab.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(ag.scalar_type() == torch::kInt8, "page_storage must be int8");
    TORCH_CHECK(ab.dim() == 3, "input must be [num_tokens, num_heads, head_dim], got ", ab.dim(), "-D");
    TORCH_CHECK(ag.dim() == 3, "page_storage must be [page_size, num_heads, head_dim], got ",
                ag.dim(), "-D");


    TORCH_CHECK(ag.is_contiguous(), "page_storage must be contiguous");
    TORCH_CHECK(aj >= 0, "token_offset_in_page must be >= 0, got ", aj);

    auto ac = ab.contiguous();
    TORCH_CHECK(ac.size(1) == ag.size(1),
                "input and page_storage must have the same num_heads, got ", ac.size(1),
                " and ", ag.size(1));
    TORCH_CHECK(ac.size(2) == ag.size(2),
                "input and page_storage must have the same head_dim, got ", ac.size(2),
                " and ", ag.size(2));

    int af = ac.size(0);
    int ah = ac.size(1) * ac.size(2);
    int ak = af * ah;

    TORCH_CHECK(
        aj + af <= ag.size(0),
        "quantize_new_scale_cuda: write would overflow the page (offset=", aj,
        ", num_tokens=", af, ", page_size=", ag.size(0), ")"
    );


    if (ak == 0) {
        return 1e-8;
    }

    auto ad = torch::zeros({1}, ac.options());
    reduce_max_abs_kernel<<<1, kReduceThreads>>>(ac.data_ptr<float>(), ak, ad.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reduce_max_abs_kernel launch failed: ", cudaGetErrorString(err));

    float ae = ad.item<float>();


    bool aa = std::isfinite(ae) && ae > 1e-12;
    double ai = aa ? (static_cast<double>(ae) / 127.0) : 1e-8;

    const int z = (ak + kQuantizeThreads - 1) / kQuantizeThreads;
    quantize_kernel<<<z, kQuantizeThreads>>>(
        ac.data_ptr<float>(), ag.data_ptr<int8_t>(),
        static_cast<int>(aj), af, ah, static_cast<float>(ai)
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "quantize_kernel launch failed: ", cudaGetErrorString(err));

    return ai;
}

void quantize_fixed_scale_cuda(
    torch::Tensor am, torch::Tensor aq, int64_t av, double au
) {
    TORCH_CHECK(am.is_cuda() && aq.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(am.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(aq.scalar_type() == torch::kInt8, "page_storage must be int8");
    TORCH_CHECK(am.dim() == 3, "input must be [num_tokens, num_heads, head_dim], got ", am.dim(), "-D");
    TORCH_CHECK(aq.dim() == 3, "page_storage must be [page_size, num_heads, head_dim], got ",
                aq.dim(), "-D");
    TORCH_CHECK(aq.is_contiguous(), "page_storage must be contiguous");


    TORCH_CHECK(std::isfinite(au) && au > 0.0, "scale must be a positive finite number, got ", au);
    TORCH_CHECK(av >= 0, "token_offset_in_page must be >= 0, got ", av);

    auto ao = am.contiguous();
    TORCH_CHECK(ao.size(1) == aq.size(1),
                "input and page_storage must have the same num_heads, got ", ao.size(1),
                " and ", aq.size(1));
    TORCH_CHECK(ao.size(2) == aq.size(2),
                "input and page_storage must have the same head_dim, got ", ao.size(2),
                " and ", aq.size(2));

    int ap = ao.size(0);
    int ar = ao.size(1) * ao.size(2);
    int aw = ap * ar;

    TORCH_CHECK(
        av + ap <= aq.size(0),
        "quantize_fixed_scale_cuda: write would overflow the page (offset=", av,
        ", num_tokens=", ap, ", page_size=", aq.size(0), ")"
    );


    if (aw == 0) {
        return;
    }

    const int al = (aw + kQuantizeThreads - 1) / kQuantizeThreads;
    quantize_kernel<<<al, kQuantizeThreads>>>(
        ao.data_ptr<float>(), aq.data_ptr<int8_t>(),
        static_cast<int>(av), ap, ar, static_cast<float>(au)
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "quantize_kernel launch failed: ", cudaGetErrorString(err));
}
