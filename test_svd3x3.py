import torch
from pytorch_svd3x3 import *

x = torch.randn((2, 3, 3)).cuda()
print(x)
print("=========== svd3x3 ===========")
print('\n'.join(map(str, svd3x3(x))))
print("=========== torch.linalg.svd ===========")
print(torch.linalg.svd(x))