import os

from setuptools import setup, Extension
from torch.utils import cpp_extension

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(name='lltm_cpp_aot', # arbitrary name?
      ext_modules=[cpp_extension.CppExtension('lltm_cpp_aot', [os.path.join(_src_path, f) for f in ['lltm.cpp']])], 
      cmdclass={'build_ext': cpp_extension.BuildExtension}) 

