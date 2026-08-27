from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="metagross",
    version="0.0.1",
    packages=["metagross"],
    ext_modules=[
        CUDAExtension(
            name="metagross._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/sanity_add.cu",
                "csrc/quantize.cu",
                "csrc/dequantize.cu",
                "csrc/paged_attention.cu",
                "csrc/quantize_int4.cu",
                "csrc/dequantize_int4.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
