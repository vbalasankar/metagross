#include "metagross_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda,
          "Elementwise add of two CUDA float32 tensors (Phase 0 sanity check)");
    m.def("quantize_new_scale", &quantize_new_scale_cuda,
          "Compute a fresh per-tensor scale via reduction, then INT8-quantize into a page");
    m.def("quantize_fixed_scale", &quantize_fixed_scale_cuda,
          "INT8-quantize into a page using an already-known scale (no reduction)");
    m.def("dequantize_page", &dequantize_page_cuda,
          "Dequantize one page's INT8 values back to FP32 using a given scale");
    m.def("paged_attention_committed", &paged_attention_committed_cuda,
          "Fused causal attention over committed (quantized) pages only; "
          "returns UNNORMALIZED (weighted_v, weight_sum, max) for the caller to merge with staging's own contribution");
    m.def("quantize_int4_new_scale", &quantize_int4_new_scale_cuda,
          "Stretch goal: INT4-quantize (2 values/byte) with a fresh reduction-computed scale. Standalone -- see csrc/quantize_int4.cu");
    m.def("dequantize_int4_page", &dequantize_int4_page_cuda,
          "Stretch goal: inverse of quantize_int4_new_scale. Standalone -- see csrc/dequantize_int4.cu");
}
