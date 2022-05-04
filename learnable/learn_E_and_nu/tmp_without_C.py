old_grid_C = None
old_mid_C = None

class MPMModel_without_C(nn.Module):
    def __init__(self, n_dim, n_grid, dx, dt, \
                 p_vol, p_rho, gravity):
        super(MPMModel_without_C, self).__init__()
        #~~~~~~~~~~~ Hyper-Parameters ~~~~~~~~~~~#
        # self.E, self.nu, self.mu_0, self.lambda_0 = E, nu, mu_0, lambda_0
        self.n_dim, self.n_grid, self.dx, self.dt, self.p_vol, self.p_rho, self.gravity = n_dim, n_grid, dx, dt, p_vol, p_rho, gravity
        self.inv_dx = float(n_grid)
        self.p_mass = p_vol * p_rho

    def forward(self, x, v, C, F, material, Jp, E, nu, first=False):
        mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

        resolution = (self.n_grid, self.n_grid, 3)
        batch_index = torch.zeros((x.shape[0], 1), dtype=torch.int, device=x.device)
        x_3d = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float, device=x.device) * self.dx], dim=1) # (x, y) -> (x, y, dx)

        #~~~~~~~~~~~ Compute C if we don't already know C (first iteration of simulation) ~~~~~~~~~~~#
        # * add a Z coordinate to convert to 3D, then use 3D CUDA functions by Zhiao 

        v_add = self.p_mass * v
        grid_v = mpm_p2g(x_3d, v_add, resolution, batch_index, self.dx) # [1, 2, G, G, 3]
        m_add = torch.ones((x.shape[0], 2), device=x.device, dtype=torch.float) * self.p_mass
        grid_m = mpm_p2g(x_3d, m_add, resolution, batch_index, self.dx) # [1, 2, G, G, 3]
        non_empty_mask = grid_m > 0
        grid_v[non_empty_mask] /= grid_m[non_empty_mask] # momentum to velocity
        # print(grid_v.max(), v.max())

        grid_x = torch.stack(torch.meshgrid(torch.arange(self.n_grid, device=x.device), torch.arange(self.n_grid, device=x.device), indexing='ij'), dim=-1) * self.dx # [G, G, D]
        grid_x = grid_x.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(-1) # [1, 2, G, G, 1]
        grid_C = (grid_v.unsqueeze(2) * grid_x.unsqueeze(1)).view(1, self.n_dim**2, self.n_grid, self.n_grid, 3)
        # print("new C max", grid_C.max(), "min", grid_C.max())
        global old_grid_C
        if old_grid_C is not None:
            print("grid_C_distance:", ((grid_C - old_grid_C)**2).mean())
        # print("grid_C_max:", torch.abs(grid_C).max())
        new_C = mpm_g2p(x_3d, grid_C, batch_index, self.dx) # [N, D]
        new_C = new_C.view(-1, self.n_dim, self.n_dim)
        global old_mid_C
        if old_mid_C is not None:
            print("mid_C_distance:", ((new_C - old_mid_C)**2).mean())
            print("mid_C_max:", torch.abs(new_C).max())
        new_C -= v.unsqueeze(2) * x.unsqueeze(1)
        new_C *= 4 * self.inv_dx**2

        print("C distance:", ((new_C - C) ** 2).mean())

        print(f"C[:2] = {C[:2]}, new_C[:2] = {new_C[:2]}")

        #~~~~~~~~~~~ Particle state update ~~~~~~~~~~~#
        x += self.dt * v
        x_3d = torch.cat([x, torch.ones((x.shape[0], 1), dtype=torch.float, device=x.device) * self.dx], dim=1) # (x, y) -> (x, y, dx)

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
        stress = (-self.dt * self.p_vol * 4 * self.inv_dx **2) * stress # [N, D, D]
        affine = stress + self.p_mass * C # [N, D, D]

        #~~~~~~~~~~~ Particle to grid (P2G) ~~~~~~~~~~~#

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
        
        grid_v = grid_v.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, 2, G, G]
        # ? how to generate fake grid_v?
        grid_v = torch.stack([grid_v, grid_v, grid_v], dim=-1) # [1, 2, G, G, 3]
        old_grid_v = grid_v.clone()

        new_v = mpm_g2p(x_3d, grid_v, batch_index, self.dx)

        v_add = new_v
        grid_v = mpm_p2g(x_3d, v_add, resolution, batch_index, self.dx) # [1, 2, G, G, 3]
        m_add = torch.ones((x.shape[0], 2), device=x.device, dtype=torch.float)
        grid_m = mpm_p2g(x_3d, m_add, resolution, batch_index, self.dx) # [1, 2, G, G, 3]
        # non_empty_mask = grid_m > 0
        grid_v[non_empty_mask] /= grid_m[non_empty_mask] # momentum to velocity

        print("grid_v:", grid_v)
        print("old_grid_v:", old_grid_v)
        
        grid_x = torch.stack(torch.meshgrid(torch.arange(self.n_grid, device=x.device), torch.arange(self.n_grid, device=x.device), indexing='ij'), dim=-1) * self.dx # [G, G, D]
        grid_x = grid_x.permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(-1) # [1, 2, G, G, 1]
        grid_C = (grid_v.unsqueeze(2) * grid_x.unsqueeze(1)).view(1, self.n_dim**2, self.n_grid, self.n_grid, 3)
        # global old_grid_C 
        old_grid_C = grid_C.clone()

        # print("old C max", grid_C.max())

        new_C = mpm_g2p(x_3d, grid_C, batch_index, self.dx) # [N, D]
        new_C = new_C.view(-1, self.n_dim, self.n_dim)
        old_mid_C = new_C.clone()
        new_C -= new_v.unsqueeze(2) * x.unsqueeze(1)
        new_C *= 4 * self.inv_dx**2

        v = new_v
        C = new_C
        # x += self.dt * v

        return x, v, C, F, material, Jp
