import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from functional import avg_voxelize, mpm_p2g, mpm_g2p

# from torch.profiler import profile, record_function, ProfilerActivity
# from torch_batch_svd import svd as fast_svd
from pytorch_svd3x3 import svd3x3

import random
random.seed(20010313)
np.random.seed(20010313)
torch.manual_seed(20010313)


class JellyV2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, clip_len=10, cache=True):
        self.data_dir = data_dir
        self.split = split
        self.clip_len = clip_len
        self.cache = cache

        with open(os.path.join(data_dir, 'config_dict.json'), 'r') as f:
            self.config_dict = json.load(f)
        self.traj_list = self.config_dict['splits'][split]

        if self.cache:
            self.traj_cache = {}

    def __len__(self):
        return len(self.traj_list)
    
    def __getitem__(self, traj_idx):
        ''' return x_traj, v_traj, C_traj, F_traj (length: self.clip_len + 1) '''

        if self.cache and traj_idx in self.traj_cache:
            data_dict = self.traj_cache[traj_idx]
        else:
            data_dict = torch.load(os.path.join(self.data_dir, self.traj_list[traj_idx], 'data_dict.pth'), map_location="cpu")
            if self.cache:
                self.traj_cache[traj_idx] = data_dict
        
        traj_len = len(data_dict['x_traj'])
        clip_start = np.random.randint(traj_len - self.clip_len)
        clip_end = clip_start + self.clip_len
        return data_dict['x_traj'][clip_start: clip_end + 1], \
                data_dict['v_traj'][clip_start: clip_end + 1], \
                data_dict['C_traj'][clip_start: clip_end + 1], \
                data_dict['F_traj'][clip_start: clip_end + 1]


if __name__ == '__main__':
    train_dataset = JellyV2Dataset('learnable/learn_Psi/data/jelly_v2/config_0000', 'train', clip_len=10)
    print(train_dataset.config_dict)
    print(train_dataset[0])