#include "metagross_ops.h"
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kReduceThreads = 512;
constexpr int kQuantizeThreads = 256;

__global__ void reduce_max_abs_kernel_int4(const float* b, int n, float* e) {
    __shared__ float f[kReduceThreads];
    int tid = threadIdx.x;

    float c = 0.0f;
    for (int i = tid; i < n; i += kReduceThreads) {
        c = fmaxf(c, fabsf(b[i]));
    }
    f[tid] = c;
    __syncthreads();

    for (int g = kReduceThreads / 2; g > 0; g >>= 1) {
        if (tid < g) {
            f[tid] = fmaxf(f[tid], f[tid + g]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *e = f[0];
    }
}

__global__ void quantize_int4_kernel(
    const float* l,
    uint8_t* r,
    int z,
    int o,
    int v,
    float w
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int aa = o * (v / 2);
    if (idx >= aa) return;

    int u = v / 2;
    int y = idx / u;
    int j = idx % u;
    int d0 = j * 2;
    int d1 = d0 + 1;

    int m = y * v;


    float v0 = l[m + d0];
    float v1 = l[m + d1];
    int32_t q0 = isnan(v0) ? 0 : static_cast<int32_t>(roundf(fminf(7.0f, fmaxf(-7.0f, v0 / w))));
    int32_t q1 = isnan(v1) ? 0 : static_cast<int32_t>(roundf(fminf(7.0f, fmaxf(-7.0f, v1 / w))));
    uint8_t low = static_cast<uint8_t>(q0) & 0x0F;
    uint8_t k = static_cast<uint8_t>(q1) & 0x0F;
    uint8_t q = low | (k << 4);

    int p = (z + y) * u + j;
    r[p] = q;
}

}

double quantize_int4_new_scale_cuda(torch::Tensor ad, torch::Tensor ai, int64_t al) {
    TORCH_CHECK(ad.is_cuda() && ai.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(ad.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(ai.scalar_type() == torch::kUInt8, "packed_storage must be uint8");
    TORCH_CHECK(ad.dim() == 3, "input must be [num_tokens, num_heads, head_dim], got ", ad.dim(), "-D");
    TORCH_CHECK(ai.dim() == 3, "packed_storage must be [page_size, num_heads, head_dim/2], got ",
                ai.dim(), "-D");
    TORCH_CHECK(ai.is_contiguous(), "packed_storage must be contiguous");
    TORCH_CHECK(ad.size(2) > 0, "head_dim must be positive, got ", ad.size(2));
    TORCH_CHECK(ad.size(2) % 2 == 0, "head_dim must be even for INT4 packing, got ", ad.size(2));
    TORCH_CHECK(al >= 0, "token_offset_in_page must be >= 0, got ", al);

    auto ae = ad.contiguous();
    TORCH_CHECK(ae.size(1) == ai.size(1),
                "input and packed_storage must have the same num_heads, got ", ae.size(1),
                " and ", ai.size(1));
    TORCH_CHECK(ai.size(2) == ae.size(2) / 2,
                "packed_storage's packed head_dim must be input's head_dim / 2, got ", ai.size(2),
                " for head_dim=", ae.size(2));

    int ah = ae.size(0);
    int aj = ae.size(1) * ae.size(2);
    int am = ah * aj;

    TORCH_CHECK(
        al + ah <= ai.size(0),
        "quantize_int4_new_scale_cuda: write would overflow the page (offset=", al,
        ", num_tokens=", ah, ", page_size=", ai.size(0), ")"
    );


    if (am == 0) {
        return 1e-8;
    }

    auto af = torch::zeros({1}, ae.options());
    reduce_max_abs_kernel_int4<<<1, kReduceThreads>>>(ae.data_ptr<float>(), am, af.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reduce_max_abs_kernel_int4 launch failed: ", cudaGetErrorString(err));

    float ag = af.item<float>();


    bool ac = std::isfinite(ag) && ag > 1e-12;
    double ak = ac ? (static_cast<double>(ag) / 7.0) : 1e-8;

    int ao = ah * (aj / 2);
    const int ab = (ao + kQuantizeThreads - 1) / kQuantizeThreads;
    quantize_int4_kernel<<<ab, kQuantizeThreads>>>(
        ae.data_ptr<float>(), ai.data_ptr<uint8_t>(),
        static_cast<int>(al), ah, aj, static_cast<float>(ak)
    );
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "quantize_int4_kernel launch failed: ", cudaGetErrorString(err));

    return ak;
}
