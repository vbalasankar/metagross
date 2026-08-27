#pragma once
#include <torch/extension.h>
#include <tuple>


torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);


double quantize_new_scale_cuda(torch::Tensor c, torch::Tensor d, int64_t e);

void quantize_fixed_scale_cuda(
    torch::Tensor f, torch::Tensor g, int64_t j, double i
);


torch::Tensor dequantize_page_cuda(
    torch::Tensor l, int64_t o, int64_t k, double n
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> paged_attention_committed_cuda(
    torch::Tensor v, torch::Tensor r, torch::Tensor w,
    torch::Tensor t, torch::Tensor u, torch::Tensor p,
    double q
);


double quantize_int4_new_scale_cuda(torch::Tensor x, torch::Tensor y, int64_t z);

torch::Tensor dequantize_int4_page_cuda(
    torch::Tensor ab, int64_t ad, int64_t aa, double ac
);
