#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA_v2(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_IS_INT(x)                                                        \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                             \
           #x " must be an int tensor")

#define CHECK_IS_FLOAT(x)                                                      \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                           \
           #x " must be a float tensor")
  
#define CHECK_IS_DOUBLE(x)                                                      \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Double,                           \
           #x " must be a double tensor")

#endif