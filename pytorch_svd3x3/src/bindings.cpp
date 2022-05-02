#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include "svd3x3.hpp"

PYBIND11_MODULE(svd3x3, m) {
    m.def("svd3x3_forward", &svd3x3_forward,
          "fast svd for 3x3 matrix (CUDA), forward function");
    m.def("svd3x3_backward", &svd3x3_backward,
          "fast svd for 3x3 matrix (CUDA), backward function");
}