import glob
import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from torch.utils import cpp_extension

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.realpath(this_dir)

    main_file = glob.glob(os.path.join(extensions_dir, "GPU-PyTorch-Ext.cpp"))
    source_cpu = []  # cpu implementation not available now
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    # compile CUDA implementation when available
    if torch.cuda.is_available() and CUDA_HOME is not None:
        print('a')
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    
    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    print(sources)
    
    ext_modules = [
        extension(
            "FastConv_Gpu",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="FastConv_Gpu",
    version="0.1",
    author="Amir Ofir",
    description="FastConv gpu implementation in pytorch",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)


# Guide for reasons it will not compile:
# 1. cast.h(1393): error: expression must be a pointer to a complete object type ->
#      Open cast.h
#      Replace "explicit operator type&() { return *(this->value); }" with "explicit operator type&() { *((type *)(this->value)); }""
#
# 2. module.h(483): error: a member with an in-class initializer must be const ->
#      Open c10/macros/Macros.h
#      Find #if defined(_MSC_VER) && defined(__CUDACC__)
#           #define CONSTEXPR_EXCEPT_WIN_CUDA
#           #define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__
#      Set  #define CONSTEXPR_EXCEPT_WIN_CUDA const
#      Find #if defined(_MSC_VER) && defined(__CUDACC__)
#           #define CONSTEXPR_EXCEPT_WIN_CUDA
#           #define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA
#      Set  #define CONSTEXPR_EXCEPT_WIN_CUDA const

