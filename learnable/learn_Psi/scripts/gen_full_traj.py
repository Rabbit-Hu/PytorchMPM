import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from model.model import MPMModelLearnedPhi

import random
random.seed(20010313)
np.random.seed(20010313)
torch.manual_seed(20010313)


def main(args):
    import taichi as ti # only for GUI (TODO: re-implement GUI to remove dependence on taichi)
    ti.init(arch=ti.cpu)
    gui_all = ti.GUI("All", res=512, background_color=0x112F41)
    gui_gt = ti.GUI("Pred", res=512, background_color=0x112F41, show_gui=False)
    gui_pred = ti.GUI("Pred", res=512, background_color=0x112F41, show_gui=False)
    gui_guess = ti.GUI("Guess", res=512, background_color=0x112F41, show_gui=False)

    device = torch.device('cuda:0')

    frame_dt = 2e-3
    E_range = (5e2, 20e2) # TODO: save E_range and nu_range into data
    nu_range = (0.01, 0.4)

    with open(os.path.join(args.traj_path, '..', 'config_dict.json'), 'r') as f:
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

    data_dict = torch.load(os.path.join(args.traj_path, 'data_dict.pth'), map_location="cpu")
    traj_len = len(data_dict['x_traj'])
    # mpm_model_init_params = data_dict['n_dim'], data_dict['n_grid'], 1/data_dict['n_grid'], data_dict['dt'], \
    #                         data_dict['p_vol'], data_dict['p_rho'], data_dict['gravity']
    # E_gt, nu_gt = data_dict['E'].to(device), data_dict['nu'].to(device) # on cuda:0; modify if this causes trouble
    # n_iter_per_frame = int(data_dict['frame_dt'] / data_dict['dt'])

    print(f'E_gt = {E_gt}, nu_gt = {nu_gt}')

    # E = torch.rand((1,), dtype=torch.float, device=device) * (E_range[1] - E_range[0]) + E_range[0]
    # nu = torch.rand((1,), dtype=torch.float, device=device) * (nu_range[1] - nu_range[0]) + nu_range[0]

    log_dir = os.path.join('/root/Concept/PytorchMPM/learnable/learn_Psi/log', args.exp_name, f'{os.path.split(args.traj_path)[-1]}_gen')
    video_all_dir = os.path.join(log_dir, 'video_all')
    video_gt_dir = os.path.join(log_dir, 'video_gt')
    video_pred_dir = os.path.join(log_dir, 'video_pred')
    video_guess_dir = os.path.join(log_dir, 'video_guess')
    os.makedirs(video_all_dir, exist_ok=True)
    os.makedirs(video_gt_dir, exist_ok=True)
    os.makedirs(video_pred_dir, exist_ok=True)
    os.makedirs(video_guess_dir, exist_ok=True)

    x_traj = data_dict['x_traj']
    x_phi, v_phi, C_phi, F_phi = data_dict['x_traj'][args.start_frame].to(device), data_dict['v_traj'][args.start_frame].to(device), \
                                 data_dict['C_traj'][args.start_frame].to(device), data_dict['F_traj'][args.start_frame].to(device)
    x_enu, v_enu, C_enu, F_enu = x_phi.clone(), v_phi.clone(), C_phi.clone(), F_phi.clone()
    material = torch.ones((len(x_phi),), dtype=torch.int, device=device)
    Jp = torch.ones((len(x_phi),), dtype=torch.float, device=device)

    mpm_model_learned_phi = MPMModelLearnedPhi(2, n_grid, dx, dt, p_vol, p_rho, gravity, psi_model_input_type=args.psi_model_input_type).to(device)
    mpm_model_learned_phi.load_state_dict(torch.load(args.model_path))
    mpm_model_guess = MPMModelLearnedPhi(2, n_grid, dx, dt, p_vol, p_rho, gravity, learn_phi=False, psi_model_input_type=args.psi_model_input_type).to(device)

    end_frame = args.end_frame if args.end_frame is not None else traj_len

    with torch.no_grad():
        for frame_i in range(args.start_frame, end_frame):
            print(f"frame_i = {frame_i}")
            for s in range(n_iter_per_frame):
                x_phi, v_phi, C_phi, F_phi, material, Jp = mpm_model_learned_phi(x_phi, v_phi, C_phi, F_phi, material, Jp)
                x_enu, v_enu, C_enu, F_enu, material, Jp = mpm_model_guess(x_enu, v_enu, C_enu, F_enu, material, Jp)

            gui_gt.circles(x_traj[frame_i].numpy(), radius=1.5, color=0xEEEEF0)
            gui_all.circles(x_traj[frame_i].numpy(), radius=1.5, color=0xEEEEF0)
            gui_pred.circles(x_phi.detach().cpu().numpy(), radius=1.5, color=0xED553B)
            gui_all.circles(x_phi.detach().cpu().numpy(), radius=1.5, color=0xED553B)
            gui_guess.circles(x_enu.detach().cpu().numpy(), radius=1.5, color=0x068587)
            gui_all.circles(x_enu.detach().cpu().numpy(), radius=1.5, color=0x068587)
            # NOTE: use ffmpeg to convert saved frames to video:
            #       ffmpeg -framerate 30 -pattern_type glob -i '*.png' -vcodec mpeg4 -vb 20M out.mov
            gui_gt.show(os.path.join(video_gt_dir, f"{frame_i:06d}.png"))
            gui_pred.show(os.path.join(video_pred_dir, f"{frame_i:06d}.png"))
            gui_guess.show(os.path.join(video_guess_dir, f"{frame_i:06d}.png"))
            gui_all.show(os.path.join(video_all_dir, f"{frame_i:06d}.png"))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--traj_path', type=str, default='/root/Concept/PytorchMPM/learnable/learn_Psi/data/jelly_v2/config_0000/traj_0001')
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--model_path', type=str, default='/root/Concept/PytorchMPM/learnable/learn_Psi/log/exp/traj_0000_clip_0000/model/checkpoint_0079_loss_6.48.pth')
    parser.add_argument('--psi_model_input_type', type=str, default='eigen')
    args = parser.parse_args()
    print(args)

    main(args)


# DISPLAY=:20 python learnable/learn_Psi/scripts/gen_full_traj.py --traj_path=/root/Concept/PytorchMPM/learnable/learn_Psi/data/jelly_v2/config_0000/traj_0002 --model_path=/root/Concept/PytorchMPM/learnable/learn_Psi/log/eigen_v2/traj_0000_clip_0000/model/checkpoint_0129_loss_9.84.pth --psi_model_input_type eigen --exp_name eigen_v2 --start_frame 30 --end_frame 60