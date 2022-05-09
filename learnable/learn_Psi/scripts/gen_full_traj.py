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
    import taichi as ti # only for GUI (TODO: re-implement GUI to remove dependence on taichi)
    ti.init(arch=ti.cpu)
    gui_gt = ti.GUI("Pred", res=512, background_color=0x112F41)
    gui_pred = ti.GUI("Pred", res=512, background_color=0x112F41)
    gui_guess = ti.GUI("Guess", res=512, background_color=0x112F41)

    device = torch.device('cuda:0')

    frame_dt = 2e-3
    E_range = (5e2, 20e2) # TODO: save E_range and nu_range into data
    nu_range = (0.01, 0.4)

    data_dict = torch.load(os.path.join(args.traj_path, 'data_dict.pth'), map_location="cpu")
    traj_len = len(data_dict['x_traj'])
    mpm_model_init_params = data_dict['n_dim'], data_dict['n_grid'], 1/data_dict['n_grid'], data_dict['dt'], \
                            data_dict['p_vol'], data_dict['p_rho'], data_dict['gravity']
    E_gt, nu_gt = data_dict['E'].to(device), data_dict['nu'].to(device) # on cuda:0; modify if this causes trouble
    n_iter_per_frame = int(data_dict['frame_dt'] / data_dict['dt'])

    E = torch.rand((1,), dtype=torch.float, device=device) * (E_range[1] - E_range[0]) + E_range[0]
    nu = torch.rand((1,), dtype=torch.float, device=device) * (nu_range[1] - nu_range[0]) + nu_range[0]

    log_dir = os.path.join('/root/Concept/PytorchMPM/learnable/learn_Psi/log', f'{os.path.split(args.traj_path)[-1]}_gen')
    video_dir = os.path.join(log_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)

    x_traj = data_dict['x_traj']
    x_phi, v_phi, C_phi, F_phi = data_dict['x_traj'][args.start_frame].to(device), data_dict['v_traj'][args.start_frame].to(device), \
                                 data_dict['C_traj'][args.start_frame].to(device), data_dict['F_traj'][args.start_frame].to(device)
    x_enu, v_enu, C_enu, F_enu = x_phi.clone(), v_phi.clone(), C_phi.clone(), F_phi.clone()
    material = torch.ones((len(x_phi),), dtype=torch.int, device=device)
    Jp = torch.ones((len(x_phi),), dtype=torch.float, device=device)

    mpm_model_learned_phi = MPMModelLearnedPhi(*mpm_model_init_params).to(device)
    mpm_model_learned_phi.load_state_dict(torch.load(args.model_path))
    mpm_model_guess = MPMModelLearnedPhi(*mpm_model_init_params, learn_phi=False).to(device)

    with torch.no_grad():
        for frame_i in range(args.start_frame, traj_len):
            print(f"frame_i = {frame_i}")
            for s in range(n_iter_per_frame):
                x_phi, v_phi, C_phi, F_phi, material, Jp = mpm_model_learned_phi(x_phi, v_phi, C_phi, F_phi, material, Jp, E, nu)
                x_enu, v_enu, C_enu, F_enu, material, Jp = mpm_model_learned_phi(x_enu, v_enu, C_enu, F_enu, material, Jp, E, nu)

            gui_gt.circles(x_traj[frame_i].numpy(), radius=1.5, color=0xEEEEF0)
            gui_pred.circles(x_phi.detach().cpu().numpy(), radius=1.5, color=0xED553B)
            gui_guess.circles(x_enu.detach().cpu().numpy(), radius=1.5, color=0x068587)
            filename = os.path.join(video_dir, f"{frame_i:06d}.png")
            # NOTE: use ffmpeg to convert saved frames to video:
            #       ffmpeg -framerate 30 -pattern_type glob -i '*.png' -vcodec mpeg4 -vb 20M out.mov
            gui_pred.show(os.path.join(video_dir, f"{frame_i:06d}.png")) # Change to gui.show(f'{frame:06d}.png') to write images to disk
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_path', type=str, default='learnable/learn_Psi/data/jelly/traj_0000')
    parser.add_argument('--start_frame', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='learnable/learn_Psi/log/exp/traj_0000_clip_0000/model/checkpoint_0199_loss_12.39.pth')
    args = parser.parse_args()
    print(args)

    main(args)