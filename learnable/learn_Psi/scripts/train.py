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
from datasets.jelly_v2_dataset import JellyV2Dataset

import random
random.seed(20010313)
np.random.seed(20010313)
torch.manual_seed(20010313)




def main(args):
    device = torch.device('cuda:0')

    train_dataset = JellyV2Dataset('learnable/learn_Psi/data/jelly_v2/config_0000', split='train', clip_len=args.clip_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    config_dict = train_dataset.config_dict
    n_grid = config_dict['n_grid']
    dx = 1 / n_grid
    dt = config_dict['dt']
    frame_dt = config_dict['frame_dt']
    n_iter_per_frame = int(frame_dt / dt + 0.5)
    p_vol, p_rho = config_dict['p_vol'], config_dict['p_rho']
    gravity = config_dict['gravity']
    E_range = config_dict['E_range']
    nu_range = config_dict['nu_range']

    # log_dir = os.path.join('learnable/learn_Psi/log', args.exp_name, f'{traj_name}_clip_{clip_idx:04d}')

    mpm_model = MPMModelLearnedPhi(2, n_grid, dx, dt, p_vol, p_rho, gravity).to(device)
    optimizer = torch.optim.SGD(mpm_model.parameters(), lr=args.Psi_lr)

    criterion = nn.MSELoss()
    
    for epoch in range(args.n_epoch):
        for sample_id, (x_traj, v_traj, C_traj, F_traj) in enumerate(train_loader):
            x_traj, v_traj, C_traj, F_traj = x_traj.squeeze(0).to(device), v_traj.squeeze(0).to(device), \
                                                C_traj.squeeze(0).to(device), F_traj.squeeze(0).to(device)
            ## x_traj [clip_len+1, n_particles, 2]
            material = torch.ones((x_traj.shape[1],), dtype=torch.int, device=device) # [n_particles,]
            Jp = torch.ones((x_traj.shape[1],), dtype=torch.float, device=device) # [n_particles,]
            
            x_scale = 1e3
            loss = 0
            x, v, C, F = x_traj[0], v_traj[0], C_traj[0], F_traj[0]
            for clip_frame in range(1, args.clip_len + 1):
                for s in range(n_iter_per_frame):
                    x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)
                if args.multi_frame:
                    loss += criterion(x * x_scale, x_traj[clip_frame] * x_scale)
            if not args.multi_frame:
                loss = criterion(x * x_scale, x_traj[-1] * x_scale)
            else:
                loss /= args.clip_len

            log_str = f"iter [{epoch}/{args.n_epoch}] sample[{sample_id}/{len(train_loader)}]: loss={loss.item():.4f}"
            print(log_str)
            
            if (sample_id + 1) % args.grad_accu == 0:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='train')
    parser.add_argument('--data_dir', type=str, default='/root/Concept/PytorchMPM/learnable/learn_Psi/data/jelly_v2/config_0000')
    parser.add_argument('--n_epoch', type=int, default=1000, help='number of epoch')
    parser.add_argument('--grad_accu', type=int, default=4, help='gradient accumulation step')
    parser.add_argument('--Psi_lr', type=int, default=3e-2, help='learning rate for the neural network that outputs Psi')
    parser.add_argument('--learn_F', action='store_true')
    parser.add_argument('--learn_C', action='store_true')
    parser.add_argument('--clip_len', type=int, default=10, help='number of frames in the trajectory clip')
    parser.add_argument('--n_grad_desc_iter', type=int, default=40, help='number of gradient descent iterations')
    parser.add_argument('--multi_frame', action='store_true', help='supervised by all frames of the trajectory if multi_frame==True; otherwise single (ending) frame')
    args = parser.parse_args()
    print(args)

    main(args)