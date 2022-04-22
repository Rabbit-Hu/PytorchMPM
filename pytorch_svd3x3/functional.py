import math
import torch

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from .backend import lib

__all__ = ['svd3x3']


class Svd3x3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        '''
            input: [N, 3, 3]
        '''
        n = inputs.shape[0]
        inputs = inputs.permute(1, 2, 0).contiguous()
        results = lib.svd3x3_forward(inputs) # [21, N]
        results = results.permute(1, 0).contiguous()
        U, S, V = results.split([9, 3, 9], dim=-1)
        U = U.view(n, 3, 3)
        V = V.view(n, 3, 3)

        return U, S, V

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        raise NotImplementedError

svd3x3 = Svd3x3.apply