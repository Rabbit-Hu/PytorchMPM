import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from functional import avg_voxelize, mpm_p2g, mpm_g2p

from torch.profiler import profile, record_function, ProfilerActivity
# from torch_batch_svd import svd as fast_svd
from pytorch_svd3x3 import svd3x3

import taichi as ti # only for GUI (TODO: re-implement GUI to remove dependence on taichi)
ti.init(arch=ti.gpu)

np.random.seed(998244353)


class MPMModel(nn.Module):
    def __init__(self, n_dim, n_particles, n_grid, dx, dt, \
                 p_vol, p_rho, E, nu, mu_0, lambda_0,      \
                 use_cuda_functions=True):
        super(MPMModel, self).__init__()
        #~~~~~~~~~~~ Hyper-Parameters ~~~~~~~~~~~#
        self.n_dim, self.n_particles, self.n_grid, self.dx, self.dt, self.p_vol, self.p_rho, self.E, self.nu, self.mu_0, self.lambda_0 = n_dim, n_particles, n_grid, dx, dt, p_vol, p_rho, E, nu, mu_0, lambda_0
        self.inv_dx = float(n_grid)
        self.p_mass = p_vol * p_rho
        
        self.use_cuda_functions = use_cuda_functions

    def forward(self, x, v, C, F, material, Jp):
        #~~~~~~~~~~~ Particle state update and scatter to grid (P2G) ~~~~~~~~~~~#
        base = (x * self.inv_dx - 0.5).long() # [N, D], long, map [n + 0.5, n + 1.5) to n
        fx = x * self.inv_dx - base.float() # [N, D], float in [0.5, 1.5) (distance: [-0.5, 0.5))
        # * Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # list of [N, D]
        F = F + self.dt * torch.bmm(C, F)

        # * hardening coefficient
        h = torch.exp(10 * (1. - Jp))
        h[material == 1] = 0.3 # jelly
        mu, lamda = self.mu_0 * h, self.lambda_0 * h # [N,]
        mu[material == 0] = 0.0 # liquid

        # * compute determinant J
        # U, sig, Vh = torch.linalg.svd(F) # [N, D, D], [N, D], [N, D, D]
        F_3x3 = torch.zeros((len(x), 3, 3), device=x.device, dtype=torch.float)
        F_3x3[:, :2, :2] = F
        U, sig, Vh = svd3x3(F_3x3)
        Vh = Vh.transpose(-2, -1)
        U, sig, Vh = U[:, :2, :2], sig[:, :2], Vh[:, :2, :2]
        # assert(torch.allclose(F, torch.bmm(U, torch.bmm(torch.diag_embed(sig), Vh)), atol=1e-5))
        snow_sig = sig[material == 2]
        clamped_sig = torch.clamp(snow_sig, 1 - 2.5e-2, 1 + 4.5e-3) # snow
        Jp[material == 2] *= (snow_sig / clamped_sig).prod(dim=-1)
        sig[material == 2] = clamped_sig
        J = sig.prod(dim=-1) # [N,]
        
        F[material == 0] = torch.eye(self.n_dim, dtype=torch.float, device=F.device)[None, :, :] * torch.pow(J[material == 0], 1./self.n_dim)[:, None, None] # liquid
        F[material == 2] = torch.bmm(U[material == 2], torch.bmm(torch.diag_embed(sig[material == 2]), Vh[material == 2])) # snow

        # * stress
        stress = 2 * mu[:, None, None] * torch.bmm((F - torch.bmm(U, Vh)), F.transpose(-1, -2)) + torch.eye(self.n_dim, dtype=torch.float, device=F.device)[None, :, :] * (lamda * J * (J - 1))[:, None, None]
        stress = (-self.dt * self.p_vol * 4 * self.inv_dx **2) * stress # [N, D, D]
        affine = stress + self.p_mass * C # [N, D, D]

        if self.n_dim == 2:
            if self.use_cuda_functions:
                # * add a Z coordinate to convert to 3D, then use 3D CUDA functions by Zhiao 
                resolution = (self.n_grid, self.n_grid, 3)
                batch_index = torch.zeros((x.shape[0], 1), dtype=torch.int, device=x.device)
                x_3d = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float, device=x.device) * self.dx], dim=1) # (x, y) -> (x, y, dx)

                v_add = self.p_mass * v - torch.bmm(affine, x[:, :, None]).squeeze(-1)
                grid_v = mpm_p2g(x_3d, v_add, resolution, batch_index, self.dx) # [1, 2, G, G, 3]

                # grid_affine_add = weight[:, None, None] * affine # [N, D, D] # ! but here we need 1d feature, so inflat the affine matrix
                affine_add = affine.view(-1, self.n_dim**2)
                grid_affine = mpm_p2g(x_3d, affine_add, resolution, batch_index, self.dx) # [1, 4, G, G, 3]

                m_add = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float) * self.p_mass
                grid_m = mpm_p2g(x_3d, m_add, resolution, batch_index, self.dx) # [1, 1, G, G, 3]

                # * project back to 2D (well this demo is still a 2D toy)
                grid_v = grid_v.sum(-1).squeeze(0).permute(1, 2, 0).contiguous() # [G, G, 2]
                grid_m = grid_m.sum(-1).squeeze(0).squeeze(0) # [G, G]
                grid_affine = grid_affine.sum(-1).squeeze(0).permute(1, 2, 0).contiguous().view(self.n_grid, self.n_grid, self.n_dim, self.n_dim) # [G, G, 3, 3]
                grid_x = torch.stack(torch.meshgrid(torch.arange(self.n_grid, device=x.device), torch.arange(self.n_grid, device=x.device), indexing='ij'), dim=-1) * self.dx # [G, G, D]
                grid_v += torch.matmul(grid_affine, grid_x[:, :, :, None]).squeeze(-1) # [G, G, D]
            else:
                grid_v = torch.zeros((self.n_grid, self.n_grid, self.n_dim), dtype=torch.float, device=x.device) # grid node momentum / velocity
                grid_m = torch.zeros((self.n_grid, self.n_grid), dtype=torch.float, device=x.device) # grid node mass
                grid_affine = torch.zeros((self.n_grid, self.n_grid, self.n_dim, self.n_dim), dtype=torch.float, device=x.device) # weighted sum of affines

                for i in range(3):
                    for j in range(3):
                        offset = torch.tensor([i, j], dtype=torch.long, device=x.device) # [2,]
                        weight = w[i][:, 0] * w[j][:, 1] # [N]
                        target = base + offset

                        # ! use atomic add function torch.Tensor.index_add_
                        # grid_v stores momentum (will be divided by m later)
                        grid_v_add = weight[:, None] * (self.p_mass * v - torch.bmm(affine, x[:, :, None]).squeeze(-1))
                        grid_v = grid_v.view(self.n_grid**2, self.n_dim) # [G**2, D]
                        grid_v.index_add_(0, target[:, 0] * self.n_grid + target[:, 1], grid_v_add)
                        grid_v = grid_v.view(self.n_grid, self.n_grid, self.n_dim)

                        grid_affine_add = weight[:, None, None] * affine # [N, D, D]
                        grid_affine = grid_affine.view(self.n_grid**2, self.n_dim, self.n_dim) # [G**2, D]
                        grid_affine.index_add_(0, target[:, 0] * self.n_grid + target[:, 1], grid_affine_add)
                        grid_affine = grid_affine.view(self.n_grid, self.n_grid, self.n_dim, self.n_dim)

                        grid_m_add = weight * self.p_mass
                        grid_m = grid_m.view(self.n_grid**2)
                        grid_m.index_add_(0, target[:, 0] * self.n_grid + target[:, 1], grid_m_add)
                        grid_m = grid_m.view(self.n_grid, self.n_grid)
                
                grid_x = torch.stack(torch.meshgrid(torch.arange(self.n_grid, device=x.device), torch.arange(self.n_grid, device=x.device), indexing='ij'), dim=-1) * self.dx # [G, G, D]
                grid_v += torch.matmul(grid_affine, grid_x[:, :, :, None]).squeeze(-1) # [G, G, D]
        else:
            raise NotImplementedError

        #~~~~~~~~~~~ some grid modifications ~~~~~~~~~~~#
        non_empty_mask = grid_m > 0
        grid_v[non_empty_mask] /= grid_m[non_empty_mask][:, None] # momentum to velocity
        grid_v[:, :, 1] -= self.dt * 50 # gravity

        # set velocity near boundary to 0
        torch.clamp_(grid_v[:3, :, 0], min=0)
        torch.clamp_(grid_v[-2:, :, 0], max=0)
        torch.clamp_(grid_v[:, :3, 1], min=0)
        torch.clamp_(grid_v[:, -2:, 1], max=0)

        #~~~~~~~~~~~ grid to particle (G2P) ~~~~~~~~~~~#
        
        # TODO: use the G2P extension
        if self.n_dim == 2:
            if self.use_cuda_functions:
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
                grid_C = (grid_v[:, :, None, :, :, :] * grid_x[:, None, :, :, :, :]).view(1, self.n_dim**2, self.n_grid, self.n_grid, 3)
                new_C = mpm_g2p(x_3d, grid_C, batch_index, self.dx) # [N, D]
                new_C = new_C.view(-1, self.n_dim, self.n_dim)
                new_C -= new_v[:, :, None] * x[:, None, :]
                new_C *= 4 * self.inv_dx**2
            else:
                new_v = torch.zeros_like(v)
                new_C = torch.zeros_like(C)
                for i in range(3):
                    for j in range(3):
                        offset = torch.tensor([i, j], dtype=torch.long, device=x.device) # [2,]
                        weight = w[i][:, 0] * w[j][:, 1] # [N]
                        target = base + offset
                        g_v = grid_v[target[:, 0], target[:, 1], :] # [N, D]
                        new_v += weight[:, None] * g_v # [N, D]
                        new_C += weight[:, None, None] * g_v[:, :, None] * (base + offset)[:, None, :] * self.dx # [N, D]
                new_C -= new_v[:, :, None] * x[:, None, :]
                new_C *= 4 * self.inv_dx**2
        else:
            raise NotImplementedError

        v = new_v
        C = new_C
        x += self.dt * v

        return x, v, C, F, material, Jp


def initialize(x, v, C, F, material, Jp):
    n_particles = len(x)
    group_size = n_particles // 3
    # TODO: convert into tensors to remove for loop
    for i in range(n_particles):
        x[i] = torch.tensor([np.random.rand() * 0.2 + 0.3 + 0.10 * (i // group_size), np.random.rand() * 0.2 + 0.05 + 0.32 * (i // group_size)])
        material[i] = i // group_size # 0: fluid 1: jelly 2: snow
        v[i] = torch.tensor([0, 0])
        F[i] = torch.eye(2)
        Jp[i] = 1


def main(args):
    #~~~~~~~~~~~ Hyper-Parameters ~~~~~~~~~~~#
    n_dim = 2 # 2D simulation
    quality = 1  # Use a larger value for higher-res simulations
    n_particles, n_grid = 9000 * quality**2, 128 * quality
    dx = 1 / n_grid
    # inv_dx = float(n_grid)
    dt = 1e-4 / quality
    p_vol, p_rho = (dx * 0.5)**2, 1
    # p_mass = p_vol * p_rho
    E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

    #~~~~~~~~~~~ Device ~~~~~~~~~~~#
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    #~~~~~~~~~~~ MPM particles ~~~~~~~~~~~#
    x = torch.empty((n_particles, n_dim), dtype=torch.float, device=device) # position
    v = torch.empty((n_particles, n_dim), dtype=torch.float, device=device) # velocity
    C = torch.empty((n_particles, n_dim, n_dim), dtype=torch.float, device=device) # [N, D, D], affine velocity field
    F = torch.empty((n_particles, n_dim, n_dim), dtype=torch.float, device=device) # [N, D, D], deformation gradient
    material = torch.empty((n_particles,), dtype=torch.long, device=device) # [N, ] material id, {0: liquid, 1: jelly, 2: snow}
    Jp = torch.empty((n_particles,), dtype=torch.float, device=device) # [N, ], plastic deformation

    initialize(x, v, C, F, material, Jp)

    mpm_model = MPMModel(n_dim, n_particles, n_grid, dx, dt, p_vol, p_rho, E, nu, mu_0, lambda_0, use_cuda_functions=not args.use_loop)

    x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            for s in range(int(2e-3 // dt)):
                x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    print()
    last_time = time.time()
    frame_cnt = 0
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(2e-3 // dt)):
            x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)
        frame_cnt += 1
        if frame_cnt % 10 == 0:
            delta_time = time.time() - last_time
            last_time = time.time()
            print(f"\033[FFPS: {10/delta_time}")
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
        gui.circles(x.cpu().numpy(), radius=1.5, color=colors[material.cpu().numpy()])
        gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch MPM demo')
    parser.add_argument('--use_loop', action="store_true", default=False, help='use loop for P2G/G2P; otherwise use pytorch extensions')
    args = parser.parse_args()
    print("Args:", args)

    main(args)