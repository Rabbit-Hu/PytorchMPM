import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='lltm_cuda_aot',
    ext_modules=[
        CUDAExtension('lltm_cuda_aot', [os.path.join(_src_path, f) for f in [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu',
        ]])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })