#ifndef _SVD_HPP
#define _SVD_HPP

#include <torch/torch.h>
#include <vector>

at::Tensor svd3x3_forward(const at::Tensor input);
at::Tensor svd3x3_backward(const std::vector<at::Tensor>& grads,
                           const at::Tensor& self,
                           bool some,
                           bool compute_uv,
                           const at::Tensor& raw_u,
                           const at::Tensor& sigma,
                           const at::Tensor& raw_v);

#endif