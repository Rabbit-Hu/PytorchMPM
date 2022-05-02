import math
import torch

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from .backend import lib

__all__ = ['svd3x3']


class Svd3x3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor):
        '''
            input: [N, 3, 3]
        '''
        n = inputs.shape[0]
        inputs = inputs.permute(1, 2, 0).contiguous() # [3, 3, N]
        results = lib.svd3x3_forward(inputs) # [21, N]
        results = results.permute(1, 0).contiguous()
        U, S, V = results.split([9, 3, 9], dim=-1)
        U = U.view(n, 3, 3)
        V = V.view(n, 3, 3)

        ctx.save_for_backward(inputs, U, S, V)

        return U, S, V

    @staticmethod
    def backward(ctx, grad_u: torch.Tensor, grad_s: torch.Tensor, grad_v: torch.Tensor):
        A, U, S, V = ctx.saved_tensors
        grad_out: torch.Tensor = lib.svd3x3_backward(
            [grad_u, grad_s, grad_v], A, True, True, U.to(A.dtype), S.to(A.dtype), V.to(A.dtype)
        )
        return grad_out

svd3x3 = Svd3x3.apply