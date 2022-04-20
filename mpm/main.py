import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
# from functional import avg_voxelize, mpm_p2g, mpm_g2p

import taichi as ti # only for GUI (TODO: re-implement GUI to remove dependence on taichi)


class MPMModel(nn.Module):
    def __init__(self, n_dim, n_particles, n_grid, dx, dt, p_vol, p_rho, E, nu, mu_0, lambda_0):
        super(MPMModel, self).__init__()
        ############ Hyper-Parameters ############
        self.n_dim, self.n_particles, self.n_grid, self.dx, self.dt, self.p_vol, self.p_rho, self.E, self.nu, self.mu_0, self.lambda_0 = n_dim, n_particles, n_grid, dx, dt, p_vol, p_rho, E, nu, mu_0, lambda_0
        self.inv_dx = float(n_grid)
        self.p_mass = p_vol * p_rho

    def forward(self, x, v, C, F, material, Jp):
        grid_v = torch.zeros((self.n_grid, self.n_grid, self.n_dim), dtype=torch.float, device=x.device) # grid node momentum / velocity
        grid_m = torch.zeros((self.n_grid, self.n_grid), dtype=torch.float, device=x.device) # grid node mass
        
        ############ Particle state update and scatter to grid (P2G) ############

        base = (x * self.inv_dx - 0.5).long() # [N, D], long, map [n + 0.5, n + 1.5) to n
        fx = x * self.inv_dx - base.float() # [N, D], float in [0.5, 1.5) (distance: [-0.5, 0.5))
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # list of [N, D]
        F = F + self.dt * torch.bmm(C, F)

        # hardening coefficient
        h = torch.exp(10 * (1. - Jp))
        h[material == 1] = 0.3 # jelly
        mu, lamda = self.mu_0 * h, self.lambda_0 * h # [N,]
        mu[material == 0] = 0.0 # liquid

        # compute determinant J
        U, sig, Vh = torch.svd(F) # [N, D, D], [N, D], [N, D, D]
        snow_sig = sig[material == 2]
        clamped_sig = torch.clamp(snow_sig, 1 - 2.5e-2, 1 + 4.5e-3) # snow
        Jp[material == 2] *= (snow_sig / clamped_sig).prod(dim=-1)
        sig[material == 2] = clamped_sig
        J = sig.prod(dim=-1) # [N,]
        
        F[material == 0] = torch.eye(self.n_dim, dtype=torch.float, device=F.device)[None, :, :] * torch.pow(J[material == 0], 1./self.n_dim)[:, None, None] # liquid
        F[material == 2] = torch.bmm(U[material == 2], torch.bmm(torch.diag_embed(sig[material == 2]), Vh[material == 2])) # snow

        # stress
        stress = 2 * mu[:, None, None] * torch.bmm((F - torch.bmm(U, Vh)), F.transpose(-1, -2)) + torch.eye(self.n_dim, dtype=torch.float, device=F.device)[None, :, :] * (lamda * J * (J - 1))[:, None, None]
        stress = (-self.dt * self.p_vol * 4 * self.inv_dx **2) * stress # [N, D, D]
        affine = stress + self.p_mass * C # [N, D, D]
        
        # TODO: use the P2G extension
        if self.n_dim == 2:
            for i in range(3):
                for j in range(3):
                    offset = torch.tensor([i, j], dtype=torch.long, device=x.device) # [2,]
                    dpos = (offset.float() - fx) * self.dx # [N, D]
                    weight = w[i][:, 0] * w[j][:, 1] # [N]
                    target = base + offset
                    assert(target.min() >= 0 and target.max() < self.n_grid)
                    grid_v[target[:, 0], target[:, 1], :] += weight[:, None] * (self.p_mass * v + torch.bmm(affine, dpos[:, :, None]).squeeze(-1)) # [N, D] * ([N, D] + [N, D]) = [N, D] # ! this grid_v is momentum
                    grid_m[target[:, 0], target[:, 1]] += weight * self.p_mass # [N,]
        else:
            raise NotImplementedError

        ############ some grid modifications ############

        non_empty_mask = grid_m > 0
        grid_v[non_empty_mask] /= grid_m[non_empty_mask][:, None] # momentum to velocity
        grid_v[:, :, 1][non_empty_mask] -= self.dt * 50 # gravity
        
        # set velocity near boundary to 0
        torch.clamp_(grid_v[:3, :, 0], min=0)
        torch.clamp_(grid_v[-3:, :, 0], max=0)
        torch.clamp_(grid_v[:, :3, 1], min=0)
        torch.clamp_(grid_v[:, -3:, 1], max=0)

        print(grid_v[:, :].min(dim=0).values.min(dim=0).values)

        ############ grid to particle (G2P) ############

        new_v = torch.zeros_like(v)
        new_C = torch.zeros_like(C)
        
        # TODO: use the G2P extension
        if self.n_dim == 2:
            for i in range(3):
                for j in range(3):
                    offset = torch.tensor([i, j], dtype=torch.long, device=x.device) # [2,]
                    dpos = (offset.float() - fx) * self.dx # [N, D]
                    weight = w[i][:, 0] * w[j][:, 1] # [N]
                    target = base + offset
                    g_v = grid_v[target[:, 0], target[:, 1], :] # [N, D]
                    new_v += weight[:, None] * g_v
                    new_C += 4 * self.inv_dx * weight[:, None, None] * g_v[:, :, None] * dpos[:, None, :]
        else:
            raise NotImplementedError

        v = new_v
        C = new_C
        x += self.dt * v

        print(x)

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


def main():
    ############ Hyper-Parameters ############
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

    ############ Device ############
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    ############ MPM particles ############
    x = torch.empty((n_particles, n_dim), dtype=torch.float, device=device) # position
    v = torch.empty((n_particles, n_dim), dtype=torch.float, device=device) # velocity
    C = torch.empty((n_particles, n_dim, n_dim), dtype=torch.float, device=device) # [N, D, D], affine velocity field
    F = torch.empty((n_particles, n_dim, n_dim), dtype=torch.float, device=device) # [N, D, D], deformation gradient
    material = torch.empty((n_particles,), dtype=torch.long, device=device) # [N, ] material id, {0: liquid, 1: jelly, 2: snow}
    Jp = torch.empty((n_particles,), dtype=torch.float, device=device) # [N, ], plastic deformation

    initialize(x, v, C, F, material, Jp)

    mpm_model = MPMModel(n_dim, n_particles, n_grid, dx, dt, p_vol, p_rho, E, nu, mu_0, lambda_0)

    x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)

    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(2e-4 // dt)):
            x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
        gui.circles(x.cpu().numpy(), radius=1.5, color=colors[material.cpu().numpy()])
        gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk
    


if __name__ == '__main__':
    main()