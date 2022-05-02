#include <c10/cuda/CUDAGuard.h>
#include "svd3x3.hpp"
#include "kernel.cuh"
#include "utils.h"

at::Tensor svd3x3_forward(const at::Tensor input) {
    CHECK_CUDA_v2(input);
    CHECK_CONTIGUOUS(input);
    CHECK_IS_FLOAT(input);

    int n = input.size(-1);

    at::Tensor result = torch::zeros(
        {21, n}, at::device(input.device()).dtype(at::ScalarType::Float));
    svd3x3Cuda(input.data_ptr<float>(), result.data_ptr<float>(), n);
    return result;
}

// Copied from
// https://github.com/KinglittleQ/torch-batch-svd/blob/master/torch_batch_svd/csrc/torch_batch_svd.cpp#L115
// https://j-towns.github.io/papers/svd-derivative.pdf
//
// This makes no assumption on the signs of sigma.
at::Tensor svd3x3_backward(const std::vector<at::Tensor>& grads,
                              const at::Tensor& self,
                              bool some,
                              bool compute_uv,
                              const at::Tensor& raw_u,
                              const at::Tensor& sigma,
                              const at::Tensor& raw_v) {
    TORCH_CHECK(
        compute_uv,
        "svd_backward: Setting compute_uv to false in torch.svd doesn't "
        "compute singular matrices, ",
        "and hence we cannot compute backward. Please use "
        "torch.svd(compute_uv=True)");

    // const at::cuda::OptionalCUDAGuard device_guard(self.device());

    // A [b, m, n]
    // auto b = self.size(0);
    auto m = self.size(1);
    auto n = self.size(2);
    auto k = sigma.size(1);
    auto gsigma = grads[1];

    auto u = raw_u;
    auto v = raw_v;
    auto gu = grads[0];
    auto gv = grads[2];

    if (!some) {
        // We ignore the free subspace here because possible base vectors cancel
        // each other, e.g., both -v and +v are valid base for a dimension.
        // Don't assume behavior of any particular implementation of svd.
        u = raw_u.narrow(2, 0, k);
        v = raw_v.narrow(2, 0, k);
        if (gu.defined()) {
            gu = gu.narrow(2, 0, k);
        }
        if (gv.defined()) {
            gv = gv.narrow(2, 0, k);
        }
    }
    auto vt = v.transpose(1, 2);

    at::Tensor sigma_term;
    if (gsigma.defined()) {
        sigma_term = u.bmm(gsigma.diag_embed()).bmm(vt);
    } else {
        sigma_term = at::zeros({1}, self.options()).expand_as(self);
    }
    // in case that there are no gu and gv, we can avoid the series of kernel
    // calls below
    if (!gv.defined() && !gu.defined()) {
        return sigma_term;
    }

    auto ut = u.transpose(1, 2);
    auto im = at::eye(m, self.options());  // work if broadcast
    auto in = at::eye(n, self.options());
    auto sigma_mat = sigma.diag_embed();
    auto sigma_mat_inv = sigma.pow(-1).diag_embed();
    auto sigma_expanded_sq = sigma.pow(2).unsqueeze(1).expand_as(sigma_mat);
    auto F = sigma_expanded_sq - sigma_expanded_sq.transpose(1, 2);
    // The following two lines invert values of F, and fills the diagonal with
    // 0s. Notice that F currently has 0s on diagonal. So we fill diagonal with
    // +inf first to prevent nan from appearing in backward of this function.
    F.diagonal(0, -2, -1).fill_(INFINITY);
    F = F.pow(-1);

    at::Tensor u_term, v_term;

    if (gu.defined()) {
        u_term =
            u.bmm(F.mul(ut.bmm(gu) - gu.transpose(1, 2).bmm(u))).bmm(sigma_mat);
        if (m > k) {
            u_term = u_term + (im - u.bmm(ut)).bmm(gu).bmm(sigma_mat_inv);
        }
        u_term = u_term.bmm(vt);
    } else {
        u_term = at::zeros({1}, self.options()).expand_as(self);
    }

    if (gv.defined()) {
        auto gvt = gv.transpose(1, 2);
        v_term = sigma_mat.bmm(F.mul(vt.bmm(gv) - gvt.bmm(v))).bmm(vt);
        if (n > k) {
            v_term = v_term + sigma_mat_inv.bmm(gvt.bmm(in - v.bmm(vt)));
        }
        v_term = u.bmm(v_term);
    } else {
        v_term = at::zeros({1}, self.options()).expand_as(self);
    }

    return u_term + sigma_term + v_term;
}