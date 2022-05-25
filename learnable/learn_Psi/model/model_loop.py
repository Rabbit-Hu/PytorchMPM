import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import argparse
import json
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
# from functional import avg_voxelize, mpm_p2g, mpm_g2p

# from torch.profiler import profile, record_function, ProfilerActivity
# from torch_batch_svd import svd as fast_svd
# from pytorch_svd3x3 import svd3x3

import random
seed = 20010313
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# 
torch.use_deterministic_algorithms(True)

class PsiModel2d(nn.Module):
    '''
        simple NLP
    '''
    def __init__(self, input_type='eigen', correcting=True, hidden_dim=16, n_hidden_layer=0, learn=True, guess_E=1000, guess_nu=0.2, base_model='fixed_corotated'):
        ''' 3d:
                input_type == 'eigen': Psi = Psi(sigma_1, sigma_2, sigma_3)
                input_type == 'coeff': Psi = Psi(tr(C), tr(CC), det(C)=J^2), C = F^TF
            2d:
                input_type == 'eigen': Psi = Psi(sigma_1, sigma_2)
                input_type == 'coeff': Psi = Psi(tr(C), det(C))
            learn: False: only use guessed E and nu
        '''
        super(PsiModel2d, self).__init__()
        assert input_type in ['eigen', 'coeff', 'basis', 'enu']
        assert (not correcting) or input_type in ['eigen', 'basis', 'enu']
        assert (not correcting) or base_model in ['neo_hookean', 'fixed_corotated']
        self.input_type = input_type
        self.correcting = correcting
        self.learn = learn
        self.base_model = base_model
        
        self.guess_E = guess_E
        self.guess_nu = guess_nu

        if input_type == 'eigen': input_dim = 2
        elif input_type == 'coeff': input_dim = 2
        elif input_type == 'basis': input_dim = 7
        elif input_type == 'enu': input_dim = 2

        if input_type == 'basis':
            self.mlp = nn.Linear(input_dim, 1)
        elif input_type in ['eigen', 'coeff']:
            # self.mlp = nn.Sequential(
            #     nn.Linear(input_dim, hidden_dim),
            #     # nn.InstanceNorm1d(hidden_dim),
            #     nn.ELU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     # nn.InstanceNorm1d(hidden_dim),
            #     nn.ELU(),
            #     # nn.Linear(hidden_dim, hidden_dim),
            #     # nn.InstanceNorm1d(hidden_dim),
            #     # nn.ELU(),
            #     # nn.Linear(hidden_dim, hidden_dim),
            #     # nn.InstanceNorm1d(hidden_dim),
            #     # nn.ELU(),
            #     nn.Linear(hidden_dim, 1),
            # )
            if n_hidden_layer == 0:
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, 1),
                )
            else:
                mlp_list = [nn.Linear(input_dim, hidden_dim), nn.ELU()]
                for _ in range(n_hidden_layer - 1):
                    mlp_list.append(nn.Linear(hidden_dim, hidden_dim))
                    mlp_list.append(nn.ELU())
                mlp_list.append(nn.Linear(hidden_dim, 1),)
                self.mlp = nn.Sequential(*mlp_list)
        elif input_type == 'enu':
            self.mlp = nn.Linear(input_dim, 1)
            E = guess_E
            nu = guess_nu
            mu = E / (2 * (1 + nu))
            la = E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters
            with torch.no_grad():
                self.mlp.weight = torch.nn.Parameter(torch.tensor([[mu, la]], requires_grad=True))
    
    def forward(self, F):
        assert(not F.isnan().any())
        C = torch.bmm(F.transpose(1, 2), F)
        assert(not C.isnan().any())
        tr_C = C[:, 0, 0] + C[:, 1, 1] # [B]
        assert(not tr_C.isnan().any())
        det_C = torch.linalg.det(C) # [B]
        if det_C.isnan().any():
            for i in range(len(det_C)):
                if det_C[i].isnan().any():
                    print(C[i])
                    print(F[i])
        assert(not det_C.isnan().any())
        
        # guessed E and nu
        E = self.guess_E
        nu = self.guess_nu
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters

        if self.input_type in ['eigen', 'basis', 'enu']:
            # print("delta = ", tr_C**2 - 4 * det_C)
            delta = tr_C**2 - 4 * det_C
            assert(not delta.isnan().any())
            delta = torch.clamp(delta, min=1e-8)
            delta = torch.sqrt(delta)
            sigma_1 = torch.sqrt(torch.clamp(0.5 * (tr_C + delta), min=1e-8)) # [B]
            sigma_2 = torch.sqrt(torch.clamp(0.5 * (tr_C - delta), min=1e-8)) # [B]
            # print("in:", sigma_1[:5], sigma_2[:5])
            # assert((torch.abs(sigma_1 - sigma_2) > 5e-5).all())
            # print(tr_C, det_C, sigma_1, sigma_2)
            # print(delta.max().item(), delta.min().item(), sigma_1.max().item(), sigma_1.min().item(), sigma_2.max().item(), sigma_2.min().item())
            # torch.clamp_(sigma_1, min=1e-1)
            # torch.clamp_(sigma_2, min=1e-1)

            assert(not sigma_1.isnan().any() and not sigma_2.isnan().any())

            if self.input_type == 'eigen':
                feat = torch.stack([sigma_1, sigma_2], dim=1) # [B, 2]
            elif self.input_type == 'basis':
                feat = torch.stack([sigma_1 + sigma_2, sigma_1**2 + sigma_2**2, sigma_1 * sigma_2, (sigma_1 * sigma_2)**2, torch.log(sigma_1) + torch.log(sigma_2), torch.log(sigma_1) * torch.log(sigma_2), torch.log(sigma_1)**2 + torch.log(sigma_2)**2], dim=1)
            elif self.input_type == 'enu':
                feat = torch.stack([((sigma_1 - 1) ** 2 + (sigma_2 - 1) ** 2), 0.5 * (sigma_1 * sigma_2 - 1) ** 2], dim=1)

            if not self.learn or (self.correcting and self.input_type != 'enu'):
                if self.base_model == 'fixed_corotated':
                    Psi_est = mu * ((sigma_1 - 1) ** 2 + (sigma_2 - 1) ** 2) + la / 2 * (sigma_1 * sigma_2 - 1) ** 2
                    assert(not Psi_est.isnan().any())
                elif self.base_model == 'neo_hookean':
                    J = sigma_1 * sigma_2
                    Psi_est = mu/2 * (sigma_1**2 + sigma_2**2 - 2) - mu * torch.log(J) + la/2 * torch.log(J)**2
                    assert(not Psi_est.isnan().any())
                if not self.learn:
                    return Psi_est
        elif self.input_type == 'coeff':
            feat = torch.stack([tr_C, det_C], dim=1) # [B, 2]
        else:
            raise NotImplementedError
        out = self.mlp(feat).squeeze(-1)
        if self.correcting and self.input_type != 'enu':
            # print("out:", out.shape, " Psi:", Psi_est.shape)
            out += Psi_est
            # out = Psi_est
        return out # [B]


class MPMModelLearnedPhi(nn.Module):
    def __init__(self, n_dim, n_grid, dx, dt, \
                 p_vol, p_rho, gravity, learn_phi=True, \
                 psi_model_input_type='eigen', guess_E=1000, guess_nu=0.2,\
                 n_hidden_layer=0, \
                 base_model='fixed_corotated'):
        super(MPMModelLearnedPhi, self).__init__()
        #~~~~~~~~~~~ Hyper-Parameters ~~~~~~~~~~~#
        # self.E, self.nu, self.mu_0, self.lambda_0 = E, nu, mu_0, lambda_0
        self.n_dim, self.n_grid, self.dx, self.dt, self.p_vol, self.p_rho, self.gravity = n_dim, n_grid, dx, dt, p_vol, p_rho, gravity
        self.inv_dx = float(n_grid)
        self.p_mass = p_vol * p_rho
        self.base_model = base_model

        self.psi_model = PsiModel2d(input_type=psi_model_input_type, hidden_dim=16, learn=learn_phi, guess_E=1000, guess_nu=0.2, n_hidden_layer=n_hidden_layer)

    def forward(self, x, v, C, F, material, Jp):
        assert(not x.isnan().any())
        assert(not v.isnan().any())
        assert(not C.isnan().any())
        assert(not F.isnan().any())
        # mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

        #~~~~~~~~~~~ Particle state update ~~~~~~~~~~~#
        base = (x * self.inv_dx - 0.5).int() # [N, D], int, map [n + 0.5, n + 1.5) to n
        fx = x * self.inv_dx - base.float() # [N, D], float in [0.5, 1.5) (distance: [-0.5, 0.5))
        # * Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # list of [N, D]
        F = F + self.dt * torch.bmm(C, F)

        # assert((torch.linalg.det(F) >= 0).all()) # * often crashes here...

        # * hardening coefficient
        # h = torch.exp(10 * (1. - Jp))
        # h[material == 1] = 0.3 # jelly
        # mu, lamda = mu_0 * h, lambda_0 * h # [N,]
        # mu[material == 0] = 0.0 # liquid

        # * compute determinant J
        # U, sig, Vh = torch.linalg.svd(F) # [N, D, D], [N, D], [N, D, D]
        # assert(not F.isnan().any())
        # F_3x3 = torch.zeros((len(x), 3, 3), device=x.device, dtype=torch.float)
        # F_3x3[:, :2, :2] = F
        # U, sig, Vh = svd3x3(F_3x3)
        # Vh = Vh.transpose(-2, -1)
        # U, sig, Vh = U[:, :2, :2], sig[:, :2], Vh[:, :2, :2]
        # assert(not U.isnan().any())
        # assert(not sig.isnan().any())
        # assert(not Vh.isnan().any())
        # sig = torch.clamp(sig, min=1e-1, max=10)
        # too_close = sig[:, 0] - sig[:, 1] < 1e-2
        # sig[too_close, :] = sig[too_close, :] + torch.tensor([[5e-3, -5e-3]], device=sig.device)
        # assert((sig[:, 0] - sig[:, 1] > 5e-3).all())
        # # print("out:", sig[:5, 0], sig[:5, 1])
        # F = torch.bmm(U, torch.bmm(torch.diag_embed(sig), Vh))

        # snow_sig = sig[material == 2]
        # clamped_sig = torch.clamp(snow_sig, 1 - 2.5e-2, 1 + 4.5e-3) # snow
        # Jp[material == 2] *= (snow_sig / clamped_sig).prod(dim=-1)
        # sig[material == 2] = clamped_sig
        # J = sig.prod(dim=-1) # [N,]
        
        # F[material == 0] = torch.eye(self.n_dim, dtype=torch.float, device=F.device).unsqueeze(0) * torch.pow(J[material == 0], 1./self.n_dim).unsqueeze(1).unsqueeze(2) # liquid
        # F[material == 2] = torch.bmm(U[material == 2], torch.bmm(torch.diag_embed(sig[material == 2]), Vh[material == 2])) # snow

        # * stress
        # stress = 2 * mu.unsqueeze(1).unsqueeze(2) * torch.bmm((F - torch.bmm(U, Vh)), F.transpose(-1, -2)) + torch.eye(self.n_dim, dtype=torch.float, device=F.device).unsqueeze(0) * (lamda * J * (J - 1)).unsqueeze(1).unsqueeze(2)
        assert(not F.isnan().any())
        with torch.enable_grad():
            F.requires_grad_()
            Psi = self.psi_model(F)
            assert(not Psi.isnan().any())
            stress = torch.autograd.grad(Psi.sum(), F, create_graph=True, allow_unused=True)[0]
        assert(not stress.isnan().any())
        stress = torch.bmm(stress, F.transpose(-1, -2))
        # print("stress.abs.max =", torch.abs(stress).max())
        stress = (-self.dt * self.p_vol * 4 * self.inv_dx **2) * stress # [N, D, D]

        # print(f"stress_max={torch.abs(stress).max()}, ", end='')

        affine = stress + self.p_mass * C # [N, D, D]

        #~~~~~~~~~~~ Particle to grid (P2G) ~~~~~~~~~~~#

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
        grid_v += torch.matmul(grid_affine, grid_x.unsqueeze(3)).squeeze(-1) # [G, G, D]

        #~~~~~~~~~~~ Grid update ~~~~~~~~~~~#
        # print(f"grid_v_max before {torch.abs(grid_v).max()}, ", end='')

        EPS = 1e-8
        non_empty_mask = grid_m > EPS
        grid_v[non_empty_mask] /= grid_m[non_empty_mask].unsqueeze(1) # momentum to velocity
        grid_v[grid_m <= EPS] = 0 #^ loop 2

        grid_v[:, :, 1] -= self.dt * self.gravity # gravity

        # set velocity near boundary to 0
        torch.clamp_(grid_v[:3, :, 0], min=0)
        torch.clamp_(grid_v[-3:, :, 0], max=0)
        torch.clamp_(grid_v[:, :3, 1], min=0)
        torch.clamp_(grid_v[:, -3:, 1], max=0)

        # print(f"after {torch.abs(grid_v).max()}")

        #~~~~~~~~~~~ Grid to particle (G2P) ~~~~~~~~~~~#
        
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

        new_x = x + self.dt * v
        new_x = torch.clamp(new_x, min=0.51 * self.dx, max=(self.n_grid - 1.51) * self.dx)

        return new_x, new_v, new_C, F, material, Jp

