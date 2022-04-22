from torch.utils.cpp_extension import load

import os
_src_path = os.path.dirname(os.path.abspath(__file__))

lib = load(name="svd3x3", 
               extra_cflags=['-O3', '-std=c++17'],
               sources=[os.path.join(_src_path, 'src', f) for f in ["bindings.cpp", "svd3x3.cpp", "kernel.cu"]], 
               verbose=True # output more info
             )