import time

import torch
from tqdm import tqdm

from lltm import *

batch_size = 16
input_features = 32
state_size = 128

device = torch.device("cuda")

X = torch.randn(batch_size, input_features, device=device)
h = torch.randn(batch_size, state_size, device=device)
C = torch.randn(batch_size, state_size, device=device)

rnn = LLTM(input_features, state_size).to(device=device)

forward = 0
backward = 0
for _ in tqdm(range(1000)):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start

    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))