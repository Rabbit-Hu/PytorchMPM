#ifndef _SVD_HPP
#define _SVD_HPP

#include <torch/torch.h>
#include <vector>

at::Tensor svd3x3_forward(const at::Tensor input);

#endif