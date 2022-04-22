#include "svd3x3.hpp"
#include "kernel.cuh"
#include "utils.h"

at::Tensor svd3x3_forward(const at::Tensor input) {
    CHECK_CUDA_v2(input);
    CHECK_CONTIGUOUS(input);
    CHECK_IS_FLOAT(input);

    int n = input.size(-1);
    
    at::Tensor result = torch::zeros({21, n}, at::device(input.device()).dtype(at::ScalarType::Float));
    svd3x3Cuda(input.data_ptr<float>(), result.data_ptr<float>(), n);
    return result;
}