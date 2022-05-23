import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import argparse
import json
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

import random
random.seed(20010313)
np.random.seed(20010313)
torch.manual_seed(20010313)
torch.cuda.manual_seed(20010313)
# 

def main(args):
    import matplotlib.pyplot as plt

    import taichi as ti # only for GUI (TODO: re-implement GUI to remove dependence on taichi)
    ti.init(arch=ti.cpu)
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41, show_gui=False)

    if args.use_loop:
        from model.model_loop import MPMModelLearnedPhi
    else:
        from model.model import MPMModelLearnedPhi

    device = torch.device('cpu') if args.cpu else torch.device('cuda:0') 

    n_clip_per_traj = 1
    clip_len = args.clip_len
    n_grad_desc_iter = args.n_grad_desc_iter
    E_lr = 1e2
    nu_lr = 1e-3
    C_lr = 0
    F_lr = 1e-2
    Psi_lr = args.Psi_lr

    #* Experiment 1-1: (sanity check) estimate E and nu from the jelly data, known F
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

    traj_list = sorted([s for s in os.listdir(data_dir) if 'traj_' in s])
    for traj_name in traj_list:
        
        data_dict = torch.load(os.path.join(data_dir, traj_name, 'data_dict.pth'), map_location="cpu")
        traj_len = len(data_dict['x_traj'])

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
            clip_start = 110 # TODO: change back
            clip_end = clip_start + clip_len
            
            with open(log_path, 'a+') as f:
                log_str = f"args={args}"
                log_str += f"\ntraj_name = {traj_name}, clip_start = {clip_start}, clip_end = {clip_end}"
                f.write(log_str + '\n')

            x_start, v_start, C_start_gt, F_start_gt = data_dict['x_traj'][clip_start].to(device), data_dict['v_traj'][clip_start].to(device), \
                        data_dict['C_traj'][clip_start].to(device), data_dict['F_traj'][clip_start].to(device)
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
            # criterion = nn.L1Loss()

            mpm_model = MPMModelLearnedPhi(2, n_grid, dx, dt, p_vol, p_rho, gravity, psi_model_input_type=args.psi_model_input_type, base_model=args.base_model).to(device)
            mpm_model.train()
            if args.optimizer == 'SGD':
                optimizer = torch.optim.SGD(mpm_model.parameters(), lr=Psi_lr)
            elif args.optimizer == 'Adam':
                optimizer = torch.optim.Adam(mpm_model.parameters(), lr=Psi_lr)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.33, patience=500, min_lr=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500,], gamma=0.1)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)

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

                optimizer.zero_grad()
                
                x, v, C, F = x_start.clone(), v_start.clone(), C_start.clone(), F_start.clone()

                loss = 0
                x_scale = 1e3
                trust_loss = True
                for clip_frame in range(clip_len):
                    # print(f"     C_max={torch.abs(C).max()}, F_max={torch.abs(F).max()}")
                    for s in range(n_iter_per_frame):
                        x, v, C, F, material, Jp = mpm_model(x, v, C, F, material, Jp)
                        # print(f"s={s:02d} C_max={torch.abs(C).max()}, F_max={torch.abs(F).max()}")
                    if clip_frame == 0:
                        continue

                    if (clip_frame + 1) % args.supervise_frame_interval == 0:
                        frame_loss = criterion(x * x_scale, x_traj[clip_frame] * x_scale)
                        loss += frame_loss

                        if not trust_loss:
                            continue

                        frame_loss.backward(retain_graph=True)

                        #* calculate grad norm
                        norm_type = 2
                        parameters = [p for p in mpm_model.parameters() if p.grad is not None]
                        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

                        if grad_norm > args.grad_eps * ((clip_frame + 1) // args.supervise_frame_interval) or clip_frame + 1 == clip_len:
                            for p in mpm_model.parameters():
                                if p.grad is not None:
                                    p.grad /= (clip_frame + 1)
                            # break
                            trust_loss = False
                            print(f"stop trusting at frame {clip_frame + 1}")

   
                #     if (clip_frame + 1) % args.supervise_frame_interval == 0:
                #         loss += criterion(x * x_scale, x_traj[clip_frame] * x_scale)
                loss /= clip_len // args.supervise_frame_interval

                # loss.backward()

                if args.compare_grads:
                    #~ save torch grad 
                    torch_grads = []
                    for i, layer in enumerate(mpm_model.psi_model.mlp):
                        if isinstance(layer, nn.Linear):
                            torch_grads.append(layer.weight.grad.view(-1))
                            if i != len(mpm_model.psi_model.mlp) - 1: # the bias of the last layer has none grad
                                torch_grads.append(layer.bias.grad.view(-1))
                    # print([g.shape for g in torch_grads])

                    eps_norm = 1e-1
                    eps_len = sum([len(g) for g in torch_grads])
                    diff_grads = torch.zeros(eps_len, device=device)

                    for eps_pos in range(eps_len):
                        eps = torch.zeros(eps_len, device=device)
                        eps[eps_pos] = eps_norm      
                        
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
                    
                        diff_grads[eps_pos] = (new_loss.item() - loss.item()) / eps_norm

                        print(f"diff_grads[{eps_pos}] = {diff_grads[eps_pos]}, torch_grads[{eps_pos}] = {torch.cat(torch_grads)[eps_pos]}")
                    

                #~~~ clip grad ~~~#
                torch.nn.utils.clip_grad_norm_(mpm_model.parameters(), args.clip_grad)

                optimizer.step()
                # scheduler.step(loss.item())
                scheduler.step()

                losses.append(loss.item())

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
                    log_str = f"iter [{grad_desc_idx}/{n_grad_desc_iter}]: "
                    log_str += f"lr={scheduler.get_last_lr()[0]:.1e}, "
                    log_str += f"loss={loss.item():.3e}, E={E.item():.2f}, E_gt={E_gt:.2f}; nu={nu.item():.4f}, nu_gt={nu_gt:.4f}; C_dist={((C - C_traj[-1])**2).sum(-1).sum(-1).mean(0).item():.2f}; F_dist={((F - F_traj[-1])**2).sum(-1).sum(-1).mean(0).item():.2f}"
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
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--Psi_lr', type=float, default=3e-2)
    parser.add_argument('--learn_F', action='store_true')
    parser.add_argument('--learn_C', action='store_true')
    parser.add_argument('--clip_len', type=int, default=100, help='number of frames in the trajectory clip')
    parser.add_argument('--supervise_frame_interval', type=int, default=10)
    parser.add_argument('--n_grad_desc_iter', type=int, default=500, help='number of gradient descent iterations')
    parser.add_argument('--psi_model_input_type', type=str, default='eigen')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--base_model', type=str, default='fixed_corotated', choices=['neo_hookean', 'fixed_corotated'])
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--force_convex', action='store_true')
    parser.add_argument('--compare_grads', action='store_true')
    parser.add_argument('--use_loop', action='store_true')
    parser.add_argument('--clip_grad', type=float, default=float('inf'))
    parser.add_argument('--grad_eps', type=float, default=float('inf'))
    args = parser.parse_args()
    print(args)

    main(args)