import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json

import torch

from model.model_loop import MPMModelLearnedPhi


device = torch.device('cuda:0') 

model_path = '/root/Concept/PytorchMPM/learnable/learn_Psi/log/loop_0layer_clip_sgd_gradeps0.2_lr3/traj_0000_clip_0000/model/checkpoint_0499.pth'
data_dir = '/xiaodi-fast-vol/PytorchMPM/learnable/learn_Psi/data/jelly_v2_every_iter/config_0000'

with open(os.path.join(data_dir, 'config_dict.json'), 'r') as f:
    config_dict = json.load(f)
n_grid = config_dict['n_grid']
dx = 1 / n_grid
dt = config_dict['dt']
frame_dt = config_dict['frame_dt']
n_iter_per_frame = int(frame_dt / dt + 0.5)
p_vol, p_rho = config_dict['p_vol'], config_dict['p_rho']
gravity = config_dict['gravity']
E_range = config_dict['E_range']
nu_range = config_dict['nu_range']
E_gt = config_dict['E']
nu_gt = config_dict['nu']
E_range = config_dict['E_range']
nu_range = config_dict['nu_range']

mpm_model = MPMModelLearnedPhi(2, n_grid, dx, dt, p_vol, p_rho, gravity, psi_model_input_type='eigen', base_model='fixed_corotated').to(device)
mpm_model.load_state_dict(torch.load(model_path))

print(mpm_model.psi_model.mlp[0].weight, mpm_model.psi_model.mlp[0].bias)