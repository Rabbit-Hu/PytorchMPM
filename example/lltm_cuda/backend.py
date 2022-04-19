from torch.utils.cpp_extension import load

import os
_src_path = os.path.dirname(os.path.abspath(__file__))

lltm_cuda_jit = load(name="lltm_cuda", 
                    sources=[os.path.join(_src_path, f) for f in ["lltm_cuda.cpp", "lltm_cuda_kernel.cu"]], 
                    verbose=True # output more info
               )