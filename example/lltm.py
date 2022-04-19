import math
import torch

import os
import sys


###### Ahead of Time (CPP) ######
# import lltm_cpp_aot as lltm

###### Just in Time (CPP) ######
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lltm_cpp'))
# from backend import lltm_cpp_jit as lltm

###### Ahead of Time (CUDA) ######
# import lltm_cuda_aot as lltm

###### Just in Time (CUDA) ######
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lltm_cuda'))
from backend import lltm_cuda_jit as lltm


class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs[:5]
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)