import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import argparse

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


class MPMModel(nn.Module):
    def __init__(self, n_dim, n_grid, dx, dt, \
                 p_vol, p_rho, gravity):
        super(MPMModel, self).__init__()
        #~~~~~~~~~~~ Hyper-Parameters ~~~~~~~~~~~#
        # self.E, self.nu, self.mu_0, self.lambda_0 = E, nu, mu_0, lambda_0
        self.n_dim, self.n_grid, self.dx, self.dt, self.p_vol, self.p_rho, self.gravity = n_dim, n_grid, dx, dt, p_vol, p_rho, gravity
        self.inv_dx = float(n_grid)
        self.p_mass = p_vol * p_rho

    def forward(self, x, v, C, F, material, Jp, E, nu):
        mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

        #~~~~~~~~~~~ Particle state update ~~~~~~~~~~~#
        base = (x * self.inv_dx - 0.5).int() # [N, D], int, map [n + 0.5, n + 1.5) to n
        fx = x * self.inv_dx - base.float() # [N, D], float in [0.5, 1.5) (distance: [-0.5, 0.5))
        # * Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # list of [N, D]
        F = F + self.dt * torch.bmm(C, F)

        # * hardening coefficient
        h = torch.exp(10 * (1. - Jp))
        # h[material == 1] = 0.3 # jelly
        mu, lamda = mu_0 * h, lambda_0 * h # [N,]
        # mu[material == 0] = 0.0 # liquid

        # * compute determinant J
        # U, sig, Vh = torch.linalg.svd(F) # [N, D, D], [N, D], [N, D, D]
        F_3x3 = torch.zeros((len(x), 3, 3), device=x.device, dtype=torch.float)
        F_3x3[:, :2, :2] = F
        U, sig, Vh = svd3x3(F_3x3)
        Vh = Vh.transpose(-2, -1)
        U, sig, Vh = U[:, :2, :2], sig[:, :2], Vh[:, :2, :2]

        # snow_sig = sig[material == 2]
        # clamped_sig = torch.clamp(snow_sig, 1 - 2.5e-2, 1 + 4.5e-3) # snow
        # Jp[material == 2] *= (snow_sig / clamped_sig).prod(dim=-1)
        # sig[material == 2] = clamped_sig
        J = sig.prod(dim=-1) # [N,]
        
        # F[material == 0] = torch.eye(self.n_dim, dtype=torch.float, device=F.device).unsqueeze(0) * torch.pow(J[material == 0], 1./self.n_dim).unsqueeze(1).unsqueeze(2) # liquid
        # F[material == 2] = torch.bmm(U[material == 2], torch.bmm(torch.diag_embed(sig[material == 2]), Vh[material == 2])) # snow

        # * stress
        stress = 2 * mu.unsqueeze(1).unsqueeze(2) * torch.bmm((F - torch.bmm(U, Vh)), F.transpose(-1, -2)) + torch.eye(self.n_dim, dtype=torch.float, device=F.device).unsqueeze(0) * (lamda * J * (J - 1)).unsqueeze(1).unsqueeze(2)
        print(stress)
        stress = (-self.dt * self.p_vol * 4 * self.inv_dx **2) * stress # [N, D, D]
        affine = stress + self.p_mass * C # [N, D, D]

        #~~~~~~~~~~~ Particle to grid (P2G) ~~~~~~~~~~~#

        # * add a Z coordinate to convert to 3D, then use 3D CUDA functions by Zhiao 
        resolution = (self.n_grid, self.n_grid, 3)
        batch_index = torch.zeros((x.shape[0], 1), dtype=torch.int, device=x.device)
        x_3d = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float, device=x.device) * self.dx], dim=1) # (x, y) -> (x, y, dx)

        v_add = self.p_mass * v - torch.bmm(affine, x.unsqueeze(2)).squeeze(-1)
        grid_v = mpm_p2g(x_3d, v_add, resolution, batch_index, self.dx) # [1, 2, G, G, 3]

        # grid_affine_add = weight.unsqueeze(1).unsqueeze(2) * affine # [N, D, D] # ! but here we need 1d feature, so inflat the affine matrix
        affine_add = affine.view(-1, self.n_dim**2)
        grid_affine = mpm_p2g(x_3d, affine_add, resolution, batch_index, self.dx) # [1, 4, G, G, 3]

        m_add = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float) * self.p_mass
        grid_m = mpm_p2g(x_3d, m_add, resolution, batch_index, self.dx) # [1, 1, G, G, 3]

        # * project back to 2D (well this demo is still a 2D toy)
        grid_v = grid_v.sum(-1).squeeze(0).permute(1, 2, 0).contiguous() # [G, G, 2]
        grid_m = grid_m.sum(-1).squeeze(0).squeeze(0) # [G, G]
        grid_affine = grid_affine.sum(-1).squeeze(0).permute(1, 2, 0).contiguous().view(self.n_grid, self.n_grid, self.n_dim, self.n_dim) # [G, G, 3, 3]
        grid_x = torch.stack(torch.meshgrid(torch.arange(self.n_grid, device=x.device), torch.arange(self.n_grid, device=x.device), indexing='ij'), dim=-1) * self.dx # [G, G, D]
        grid_v += torch.matmul(grid_affine, grid_x.unsqueeze(3)).squeeze(-1) # [G, G, D]

        #~~~~~~~~~~~ Grid update ~~~~~~~~~~~#
        non_empty_mask = grid_m > 0
        grid_v[non_empty_mask] /= grid_m[non_empty_mask].unsqueeze(1) # momentum to velocity
        grid_v[:, :, 1] -= self.dt * self.gravity # gravity

        # set velocity near boundary to 0
        torch.clamp_(grid_v[:3, :, 0], min=0)
        torch.clamp_(grid_v[-3:, :, 0], max=0)
        torch.clamp_(grid_v[:, :3, 1], min=0)
        torch.clamp_(grid_v[:, -3:, 1], max=0)

        #~~~~~~~~~~~ Grid to particle (G2P) ~~~~~~~~~~~#
        
        # * add a Z coordinate to convert to 3D, then use 3D CUDA functions by Zhiao 
        resolution = (self.n_grid, self.n_grid, 3)
        batch_index = torch.zeros((x.shape[0], 1), dtype=torch.int, device=x.device)
        x_3d = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float, device=x.device) * self.dx], dim=1) # (x, y) -> (x, y, dx)
        
        grid_v = grid_v.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, 2, G, G]
        # ? how to generate fake grid_v?
        grid_v = torch.stack([grid_v, grid_v, grid_v], dim=-1) # [1, 2, G, G, 3]

        new_v = mpm_g2p(x_3d, grid_v, batch_index, self.dx)
        
        grid_x = torch.stack(torch.meshgrid(torch.arange(self.n_grid, device=x.device), torch.arange(self.n_grid, device=x.device), indexing='ij'), dim=-1) * self.dx # [G, G, D]
        grid_x = grid_x.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(-1) # [1, 2, G, G, 1]
        grid_C = (grid_v.unsqueeze(2) * grid_x.unsqueeze(1)).view(1, self.n_dim**2, self.n_grid, self.n_grid, 3)
        new_C = mpm_g2p(x_3d, grid_C, batch_index, self.dx) # [N, D]
        new_C = new_C.view(-1, self.n_dim, self.n_dim)
        new_C -= new_v.unsqueeze(2) * x.unsqueeze(1)
        new_C *= 4 * self.inv_dx**2

        return x + self.dt * v, new_v, new_C, F, material, Jp


def main():
    import taichi as ti # only for GUI (TODO: re-implement GUI to remove dependence on taichi)
    ti.init(arch=ti.gpu)
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)

    device = torch.device('cuda:0')

    n_clip_per_traj = 1
    clip_len = 1
    n_grad_desc_iter = 100
    grad_desc_lr = 1e-3
    
    frame_dt = 2e-3 # TODO: save frame_dt into data

    #* Experiment 1-1: (sanity check) estimate E and nu from the jelly data, known F
    data_dir = '/xiaodi-fast-vol/PytorchMPM/learnable/learn_E_and_nu/data/jelly'
    traj_list = os.listdir(data_dir)
    for traj_name in traj_list:
        data_dict = torch.load(os.path.join(data_dir, traj_name, 'data_dict.pth'), map_location="cpu")
        print(data_dict)
        traj_len = len(data_dict['x_traj'])
        mpm_model = MPMModel(data_dict['n_dim'], data_dict['n_grid'], 1/data_dict['n_grid'], data_dict['dt'], \
                             data_dict['p_vol'], data_dict['p_rho'], data_dict['gravity'])
        mpm_model = mpm_model.to(device)
        E_gt, nu_gt = data_dict['E'].to(device), data_dict['nu'].to(device) # on cuda:0; modify if this causes trouble

        for clip_idx in range(n_clip_per_traj):
            #* get a random clip
            clip_start = np.random.randint(traj_len - clip_len)
            clip_end = clip_start + clip_len
            x, v, C, F = data_dict['x_traj'][clip_start].to(device), data_dict['v_traj'][clip_start].to(device), \
                         data_dict['C_traj'][clip_start].to(device), data_dict['F_traj'][clip_start].to(device)
            x_end, v_end, C_end, F_end = data_dict['x_traj'][clip_end].to(device), data_dict['v_traj'][clip_end].to(device), \
                                         data_dict['C_traj'][clip_end].to(device), data_dict['F_traj'][clip_end].to(device) 
            material = torch.ones((len(x),), dtype=torch.int, device=device)
            Jp = torch.ones((len(x),), dtype=torch.float, device=device)
            
            E_range = (5e2, 20e2)
            nu_range = (0.01, 0.4)
            E = torch.rand((1,), dtype=torch.float, device=device) * (E_range[1] - E_range[0]) + E_range[0]
            nu = torch.rand((1,), dtype=torch.float, device=device) * (nu_range[1] - nu_range[0]) + nu_range[0]
            E.requires_grad_()
            nu.requires_grad_()

            # x.requires_grad_()
            # v.requires_grad_()
            # C.requires_grad_()
            # F.requires_grad_()

            print(f"init E = {E}, nu = {nu}")

            criterion = nn.MSELoss()

            for grad_desc_idx in range(n_grad_desc_iter):
                if E.grad is not None: E.grad.zero_()
                if nu.grad is not None: nu.grad.zero_()
                # for s in range(int(frame_dt // data_dict['dt'])): # TODO: change back
                for s in range(1):
                    print(f"s = {s}")
                    x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp, E, nu)
                print("x     =", x)
                print("x_end =", x_end)
                loss = criterion(v, v_end)
                # print(f"loss.device = {loss.device}")
                loss.backward()
                E -= grad_desc_lr * E.grad
                nu -= grad_desc_lr * nu.grad

                print(f"iter [{grad_desc_idx}/{n_grad_desc_iter}]: E = {E.item()}, E_gt = {E_gt.item()}; nu = {nu.item()}, nu_gt = {nu_gt.item()}")


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()