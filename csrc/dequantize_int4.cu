#include "metagross_ops.h"
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kDequantizeThreads = 256;

__device__ __forceinline__ int sign_extend_nibble(uint8_t b) {
    return static_cast<int>(static_cast<int8_t>(b << 4) >> 4);
}

__global__ void dequantize_int4_kernel(
    const uint8_t* i,
    float* f,
    int n,
    int d,
    int j,
    float l
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int o = d * j;
    if (idx >= o) return;

    int m = idx / j;
    int c = idx % j;

    uint8_t g = i[(n + m) * j + c];
    int v0 = sign_extend_nibble(g & 0x0F);
    int v1 = sign_extend_nibble((g >> 4) & 0x0F);

    int k = j * 2;
    int e = m * k + c * 2;
    f[e] = static_cast<float>(v0) * l;
    f[e + 1] = static_cast<float>(v1) * l;
}

}

torch::Tensor dequantize_int4_page_cuda(
    torch::Tensor v, int64_t z, int64_t t, double y
) {
    TORCH_CHECK(v.is_cuda(), "packed_storage must be a CUDA tensor");
    TORCH_CHECK(v.scalar_type() == torch::kUInt8, "packed_storage must be uint8");
    TORCH_CHECK(v.dim() == 3, "packed_storage must be [page_size, num_heads, head_dim/2], got ",
                v.dim(), "-D");
    TORCH_CHECK(v.is_contiguous(), "packed_storage must be contiguous");
    TORCH_CHECK(z >= 0, "token_offset_in_page must be >= 0, got ", z);
    TORCH_CHECK(t >= 0, "num_tokens must be >= 0, got ", t);
    TORCH_CHECK(std::isfinite(y) && y > 0.0, "scale must be a positive finite number, got ", y);
    TORCH_CHECK(
        z + t <= v.size(0),
        "dequantize_int4_page_cuda: read would overflow the page (offset=", z,
        ", num_tokens=", t, ", page_size=", v.size(0), ")"
    );

    int r = v.size(1);
    int q = v.size(2);
    int w = r * q;
    int aa = static_cast<int>(t) * w;

    auto u = torch::empty(
        {t, r, q * 2}, v.options().dtype(torch::kFloat32)
    );


    if (aa == 0) {
        return u;
    }

    const int p = (aa + kDequantizeThreads - 1) / kDequantizeThreads;
    dequantize_int4_kernel<<<p, kDequantizeThreads>>>(
        v.data_ptr<uint8_t>(), u.data_ptr<float>(),
        static_cast<int>(z), static_cast<int>(t), w,
        static_cast<float>(y)
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dequantize_int4_kernel launch failed: ", cudaGetErrorString(err));

    return u;
}
