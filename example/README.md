# "LLTM" Example on Writing Cpp Extensions for Pytorch

This directory stores the code (modified a little bit) from the following tutorial on cpp extensions for pytorch:

[Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)

Files:
- lltm_cpp: cpp version
    - lltm.cpp: example source code
    - setup.py: run `python setup.py install` to install the ahead-of-time package
    - backend.py: JIT (just in time) version
- lltm_cuda: cuda version (TODO)