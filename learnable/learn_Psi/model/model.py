import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import argparse
import json
import copy

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


class PsiModel2d(nn.Module):
    '''
        simple NLP
    '''
    def __init__(self, input_type='eigen', correcting=True, hidden_dim=16, learn=True, guess_E=1000, guess_nu=0.2, base_model='fixed_corotated'):
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
        elif input_type == 'basis': input_dim =4
        elif input_type == 'enu': input_dim = 2

        if input_type == 'basis':
            self.mlp = nn.Linear(input_dim, 1)
        elif input_type in ['eigen', 'coeff']:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                # nn.InstanceNorm1d(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                # nn.InstanceNorm1d(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                # nn.InstanceNorm1d(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                # nn.InstanceNorm1d(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1),
            )
        elif input_type == 'enu':
            self.mlp = nn.Linear(input_dim, 1)
            E = guess_E
            nu = guess_nu
            mu = E / (2 * (1 + nu))
            la = E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters
            with torch.no_grad():
                self.mlp.weight = torch.nn.Parameter(torch.tensor([[mu, la]], requires_grad=True))
    
    def forward(self, F):
        C = torch.bmm(F.transpose(1, 2), F)
        tr_C = C[:, 0, 0] + C[:, 1, 1] # [B]
        det_C = torch.linalg.det(C) # [B]
        
        # guessed E and nu
        E = self.guess_E
        nu = self.guess_nu
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters

        if self.input_type in ['eigen', 'basis', 'enu']:
            # print("delta = ", tr_C**2 - 4 * det_C)
            delta = tr_C**2 - 4 * det_C
            delta = torch.clamp(delta, min=1e-8)
            delta = torch.sqrt(delta)
            sigma_1 = torch.sqrt(0.5 * (tr_C + delta)) # [B]
            sigma_2 = torch.sqrt(0.5 * (tr_C - delta)) # [B]
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
                feat = torch.stack([(sigma_1 - 1)**2 + (sigma_2 - 1)**2, sigma_1 + sigma_2, torch.log(sigma_1) + torch.log(sigma_2), torch.log(sigma_1)**2 + torch.log(sigma_2)**2], dim=1)
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
                 p_vol, p_rho, gravity, learn_phi=True, psi_model_input_type='eigen', guess_E=1000, guess_nu=0.2,\
                 base_model='fixed_corotated'):
        super(MPMModelLearnedPhi, self).__init__()
        #~~~~~~~~~~~ Hyper-Parameters ~~~~~~~~~~~#
        # self.E, self.nu, self.mu_0, self.lambda_0 = E, nu, mu_0, lambda_0
        self.n_dim, self.n_grid, self.dx, self.dt, self.p_vol, self.p_rho, self.gravity = n_dim, n_grid, dx, dt, p_vol, p_rho, gravity
        self.inv_dx = float(n_grid)
        self.p_mass = p_vol * p_rho
        self.base_model = base_model

        self.psi_model = PsiModel2d(input_type=psi_model_input_type, hidden_dim=16, learn=learn_phi, guess_E=1000, guess_nu=0.2)

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
        assert(not F.isnan().any())
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

        new_x = x + self.dt * v
        new_x = torch.clamp(new_x, min=0.51 * self.dx, max=(self.n_grid - 1.51) * self.dx)

        return new_x, new_v, new_C, F, material, Jp


def main(args):
    import matplotlib.pyplot as plt

    import taichi as ti # only for GUI (TODO: re-implement GUI to remove dependence on taichi)
    ti.init(arch=ti.cpu)
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)

    device = torch.device('cuda:0')

    n_clip_per_traj = 1
    clip_len = args.clip_len
    n_grad_desc_iter = args.n_grad_desc_iter
    E_lr = 1e2
    nu_lr = 1e-3
    C_lr = 0
    F_lr = 1e-2
    Psi_lr = args.Psi_lr

    #* Experiment 1-1: (sanity check) estimate E and nu from the jelly data, known F
    data_dir = '/xiaodi-fast-vol/PytorchMPM/learnable/learn_Psi/data/jelly_v2/config_0000'

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

    traj_list = sorted([s for s in os.listdir(data_dir) if 'traj_' in s])
    for traj_name in traj_list:
        
        data_dict = torch.load(os.path.join(data_dir, traj_name, 'data_dict.pth'), map_location="cpu")
        # print(data_dict)
        traj_len = len(data_dict['x_traj'])
        # mpm_model_init_params = data_dict['n_dim'], data_dict['n_grid'], 1/data_dict['n_grid'], data_dict['dt'], \
        #                         data_dict['p_vol'], data_dict['p_rho'], data_dict['gravity']
        # E_gt, nu_gt = data_dict['E'].to(device), data_dict['nu'].to(device) # on cuda:0; modify if this causes trouble

        for clip_idx in range(n_clip_per_traj):
            log_dir = os.path.join('/root/Concept/PytorchMPM/learnable/learn_Psi/log', args.exp_name, f'{traj_name}_clip_{clip_idx:04d}')
            video_dir = os.path.join(log_dir, 'video')
            model_dir = os.path.join(log_dir, 'model')
            os.makedirs(video_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'log.txt')
            curve_path = os.path.join(log_dir, 'training_curve.png')

            #* get a random clip
            # clip_start = np.random.randint(traj_len - clip_len)
            clip_start = 11 # TODO: change back
            clip_end = clip_start + clip_len
            
            with open(log_path, 'a+') as f:
                log_str = f"args={args}"
                log_str += f"\ntraj_name = {traj_name}, clip_start = {clip_start}, clip_end = {clip_end}"
                f.write(log_str + '\n')

            x_start, v_start, C_start_gt, F_start_gt = data_dict['x_traj'][clip_start].to(device), data_dict['v_traj'][clip_start].to(device), \
                        data_dict['C_traj'][clip_start].to(device), data_dict['F_traj'][clip_start].to(device)
            # x_end, v_end, C_end, F_end = data_dict['x_traj'][clip_end].to(device), data_dict['v_traj'][clip_end].to(device), \
            #                             data_dict['C_traj'][clip_end].to(device), data_dict['F_traj'][clip_end].to(device)
            x_traj, v_traj, C_traj, F_traj = data_dict['x_traj'][clip_start + 1: clip_end + 1].to(device), \
                                             data_dict['v_traj'][clip_start + 1: clip_end + 1].to(device), \
                                             data_dict['C_traj'][clip_start + 1: clip_end + 1].to(device), \
                                             data_dict['F_traj'][clip_start + 1: clip_end + 1].to(device)

            material = torch.ones((len(x_start),), dtype=torch.int, device=device)
            Jp = torch.ones((len(x_start),), dtype=torch.float, device=device)
            
            E = torch.rand((1,), dtype=torch.float, device=device) * (E_range[1] - E_range[0]) + E_range[0]
            nu = torch.rand((1,), dtype=torch.float, device=device) * (nu_range[1] - nu_range[0]) + nu_range[0]
            print(f"init E = {E}, nu = {nu}")

            if args.learn_C:
                C_start = torch.zeros((len(x_start), 2, 2), dtype=torch.float, device=device)
            else:
                C_start = C_start_gt
            
            if args.learn_F:
                F_start = torch.eye(2, dtype=torch.float, device=device)[None, :, :].repeat(len(x_start), 1, 1)
                F_start += torch.randn_like(F_start) * 0.001
                # print("F_start =", F_start)
            else:
                F_start = F_start_gt

            criterion = nn.MSELoss()

            mpm_model = MPMModelLearnedPhi(2, n_grid, dx, dt, p_vol, p_rho, gravity, psi_model_input_type=args.psi_model_input_type, base_model=args.base_model).to(device)
            mpm_model.train()
            # mpm_model.load_state_dict(torch.load('/root/Concept/PytorchMPM/learnable/learn_Psi/log/traj_0000_clip_0000/model/checkpoint_0019_loss_158.05'))
            if args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(mpm_model.parameters(), lr=Psi_lr)
            elif args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(mpm_model.parameters(), lr=Psi_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=300, min_lr=1e-3)

            Psi_lr_decayed = Psi_lr

            last_loss = 1e18

            losses = []

            for grad_desc_idx in range(n_grad_desc_iter):
                if E.grad is not None: E.grad.zero_()
                if nu.grad is not None: nu.grad.zero_()
                if args.learn_C and C_start.grad is not None: C_start.grad.zero_()
                if args.learn_F and F_start.grad is not None: F_start.grad.zero_()

                E.requires_grad_()
                nu.requires_grad_()
                if args.learn_C: C_start.requires_grad_()
                if args.learn_F: F_start.requires_grad_()

                optimizer.zero_grad()
                
                x, v, C, F = x_start, v_start, C_start, F_start

                loss = 0
                x_scale = 1e3
                for clip_frame in range(clip_len):
                    for s in range(n_iter_per_frame):
                        x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)
                    if not args.single_frame:
                        loss += criterion(x * x_scale, x_traj[clip_frame] * x_scale)
                if args.single_frame:
                    loss = criterion(x * x_scale, x_traj[clip_len - 1] * x_scale)
                else:
                    loss /= clip_len
                # loss = criterion(v, v_end)

                # if (loss.item() > last_loss or loss.item() < 0.9 * last_loss) and Psi_lr_decayed > 1e-3:
                #     Psi_lr_decayed *= 0.2
                #     print(f"Psi_lr_decayed = {Psi_lr_decayed}")
                #     for g in optimizer.param_groups:
                #         g['lr'] = Psi_lr_decayed
                # last_loss = loss.item()

                # loss.backward(retain_graph=True)
                loss.backward()
                # print("E.data =", E.data, "E.grad.data =", E.grad.data, "E_lr * E.grad.data =", E_lr * E.grad.data)
                # print("F_start.data =", F_start.data, "F_start.grad.data =", F_start.grad.data, "F_lr * F_start.grad.data =", F_lr * F_start.grad.data)

                #~ save torch grad 
                torch_grads = []
                for i, layer in enumerate(mpm_model.psi_model.mlp):
                    if isinstance(layer, nn.Linear):
                        torch_grads.append(layer.weight.grad.view(-1))
                        if i != len(mpm_model.psi_model.mlp) - 1: # the bias of the last layer has none grad
                            torch_grads.append(layer.bias.grad.view(-1))
                print([g.shape for g in torch_grads])

                eps_norm = 1e-3
                eps = torch.randn(sum([len(g) for g in torch_grads]), device=device)
                eps *= eps_norm / torch.linalg.norm(eps)
                # print(eps)
                
                model_copy = copy.deepcopy(mpm_model)
                with torch.no_grad():
                    cur = 0
                    for i, layer in enumerate(model_copy.psi_model.mlp):
                        if isinstance(layer, nn.Linear):
                            w_shape = layer.weight.shape
                            w_size = np.prod(w_shape)
                            layer.weight += eps[cur: cur + w_size].view(*w_shape)
                            cur += w_size
                            if i != len(mpm_model.psi_model.mlp) - 1: # the bias of the last layer has none grad
                                b_shape = layer.bias.shape
                                b_size = np.prod(b_shape)
                                layer.bias += eps[cur: cur + b_size].view(*b_shape)
                                cur += b_size
                    
                    x, v, C, F = x_start, v_start, C_start, F_start
                    new_loss = 0
                    x_scale = 1e3
                    for clip_frame in range(clip_len):
                        for s in range(n_iter_per_frame):
                            x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)
                        if not args.single_frame:
                            new_loss += criterion(x * x_scale, x_traj[clip_frame] * x_scale)
                    if args.single_frame:
                        new_loss = criterion(x * x_scale, x_traj[clip_len - 1] * x_scale)
                    else:
                        new_loss /= clip_len
                    
                    

                optimizer.step()
                scheduler.step(loss.item())

                losses.append(loss.item())

                with torch.no_grad():
                # if True:
                    if args.force_convex:
                        first = True
                        for layer in mpm_model.psi_model.mlp:
                            if isinstance(layer, nn.Linear):
                                if first:
                                    first = False
                                    continue
                                layer.weight = torch.nn.Parameter(torch.abs(layer.weight))
                                # print(layer.weight.data)
                                # print(layer.weight.grad)
                                # for name, data in layer.named_parameters():
                                #     if 'weight' in name:
                                #         # print(name, data.shape)
                                #         data.clamp_(min=0.)

                with torch.no_grad():
                    # E = E - E_lr * E.grad
                    # torch.clamp_(E, min=E_range[0], max=E_range[1])
                    # nu = nu - nu_lr * nu.grad
                    # torch.clamp_(nu, min=nu_range[0], max=nu_range[1])
                    if args.learn_C:
                        C_start = C_start - C_lr * C_start.grad
                    if args.learn_F:
                        F_start = F_start - F_lr * F_start.grad

                if args.psi_model_input_type == 'basis':
                    for p in mpm_model.psi_model.mlp.parameters():
                        print(p.data)

                # print("E.data =", E.data)
                # colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
                gui.circles(x_start.detach().cpu().numpy(), radius=1.5, color=0x068587)
                gui.circles(x.detach().cpu().numpy(), radius=1.5, color=0xED553B)
                gui.circles(x_traj[-1].detach().cpu().numpy(), radius=1.5, color=0xEEEEF0)
                filename = os.path.join(video_dir, f"{grad_desc_idx:06d}.png")
                # NOTE: use ffmpeg to convert saved frames to video:
                #       ffmpeg -framerate 5 -pattern_type glob -i '*.png' -vcodec mpeg4 -vb 20M out.mov
                #       ffmpeg -framerate 5 -pattern_type glob -i 'video_all/*.png' -vcodec mpeg4 -acodec aac video_all.mov
                # Stacking videos:
                #       ffmpeg -i video_guess.mov  -i video_gt.mov -i video_pred.mov -filter_complex hstack=inputs=3 -vb 20M hstack.mp4
                gui.show(filename) # Change to gui.show(f'{frame:06d}.png') to write images to disk
                # gui.show()

                fig, ax = plt.subplots()
                ax.plot(np.arange(len(losses)), np.array(losses))
                fig.savefig(curve_path)
                plt.close(fig)

                with open(log_path, 'a+') as f:
                    log_str = f"iter [{grad_desc_idx}/{n_grad_desc_iter}]: lr={optimizer.param_groups[0]['lr']:.0e}, loss={loss.item():.4f}, E={E.item():.2f}, E_gt={E_gt:.2f}; nu={nu.item():.4f}, nu_gt={nu_gt:.4f}; C_dist={((C - C_traj[-1])**2).sum(-1).sum(-1).mean(0).item():.2f}; F_dist={((F - F_traj[-1])**2).sum(-1).sum(-1).mean(0).item():.2f}"
                    # if args.learn_C:
                    #     log_str += f"\nC_start[:-2] = {C_start[:-2]}, C_start_gt[:-2] = {C_start_gt[:-2]}"
                    # if args.learn_F:
                    #     log_str += f"\nF_start[:-2] = {F_start[:-2]}, F_start_gt[:-2] = {F_start_gt[:-2]}"
                    print(log_str)
                    f.write(log_str + '\n')
                
                if (grad_desc_idx + 1) % args.save_interval == 0:
                    torch.save(mpm_model.state_dict(), os.path.join(model_dir, f'checkpoint_{grad_desc_idx:04d}.pth'))


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--Psi_lr', type=float, default=3e-2)
    parser.add_argument('--learn_F', action='store_true')
    parser.add_argument('--learn_C', action='store_true')
    parser.add_argument('--clip_len', type=int, default=10, help='number of frames in the trajectory clip')
    parser.add_argument('--n_grad_desc_iter', type=int, default=500, help='number of gradient descent iterations')
    parser.add_argument('--single_frame', action='store_true', help='supervised by the single (ending) frame of the trajectory if single_frame==True; otherwise supervised by all the frames')
    parser.add_argument('--psi_model_input_type', type=str, default='eigen')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--base_model', type=str, default='fixed_corotated', choices=['neo_hookean', 'fixed_corotated'])
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--force_convex', action='store_true')
    args = parser.parse_args()
    print(args)

    main(args)