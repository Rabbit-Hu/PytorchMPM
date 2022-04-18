from torch.utils.cpp_extension import load

import os
_src_path = os.path.dirname(os.path.abspath(__file__))

lltm_cpp_jit = load(name="lltm_cpp", 
                    sources=[os.path.join(_src_path, f) for f in ["lltm.cpp"]], 
                    verbose=True # output more info
               )