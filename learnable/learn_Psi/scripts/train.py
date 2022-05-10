import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from functional import avg_voxelize, mpm_p2g, mpm_g2p

# from torch.profiler import profile, record_function, ProfilerActivity
# from torch_batch_svd import svd as fast_svd
from pytorch_svd3x3 import svd3x3

from model.model import MPMModelLearnedPhi

import random
random.seed(20010313)
np.random.seed(20010313)
torch.manual_seed(20010313)


def main(args):
    pass# TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='train')
    parser.add_argument('--data_dir', type=str, default='/root/Concept/PytorchMPM/learnable/learn_Psi/data/jelly_v2/config_0000')
    parser.add_argument('--learn_F', action='store_true')
    parser.add_argument('--learn_C', action='store_true')
    parser.add_argument('--clip_len', type=int, default=10, help='number of frames in the trajectory clip')
    parser.add_argument('--n_grad_desc_iter', type=int, default=40, help='number of gradient descent iterations')
    parser.add_argument('--multi_frame', action='store_true', help='supervised by all frames of the trajectory if multi_frame==True; otherwise single (ending) frame')
    args = parser.parse_args()
    print(args)

    main(args)