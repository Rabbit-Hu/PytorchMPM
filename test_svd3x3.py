from pytorch_svd3x3 import svd3x3
import torch

A_torch = torch.randn(2, 3, 3).cuda()
A_33 = A_torch.clone()
A_torch.requires_grad_()
A_33.requires_grad_()
print(A_torch.is_leaf, A_33.is_leaf)
U_torch, sigma_torch, V_torch = torch.svd(A_torch)
U_33, sigma_33, V_33 = svd3x3(A_33)

print("====== U ======")
print("U_torch:", U_torch)
print("U_33   :", U_33)

print("====== sigma ======")
print("sigma_torch:", sigma_torch)
print("sigma_33   :", sigma_33)

print("====== V ======")
print("V_torch:", V_torch)
print("V_33   :", V_33)


print("====== A.grad ======")

U_c, sigma_c, V_c = torch.randn(2, 3, 3).cuda(), torch.randn(2, 3).cuda(), torch.randn(2, 3, 3).cuda()

loss_torch = (U_c * U_torch**2).sum() + (sigma_c * sigma_torch**2).sum() + (V_c * V_torch**2).sum()
loss_torch.backward()
print("loss_torch:", A_torch.grad)

loss_33 = (U_c * U_33**2).sum() + (sigma_c * sigma_33**2).sum() + (V_c * V_33**2).sum()
loss_33.backward()
print("loss_torch:", A_33.grad)
