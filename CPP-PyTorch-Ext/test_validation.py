import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
import timeit
import time
from gck_layer import GCK3x3Layer

kernel_dim = 3

def tensors_equal(a,b):
    b = torch.allclose(a, b, atol=0.01)
    if (b):
        print('same: True')
    else:
        print('Same: False (diff:', ((a-b).max()), ')')

def compareResults(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
    input = torch.randint(0,100, (batch_size, in_channels, input_dim, input_dim), dtype=torch.float32)
    kernels = torch.ones(out_channels, in_channels, kernel_dim, kernel_dim, dtype=torch.float32)
    expected = F.conv2d(input, kernels)

    layer = GCK3x3Layer(in_channels, out_channels, 3, False, input_dim - 2, kernels)
    result = layer.forward(input)

    tensors_equal(expected, result)


lst = [
    (1,1,8,1024),
    (1,1,16,256),
    (1,1,64,512),
    (1,1,128,512),
    (1,8,16,128)
    ]

for batch_size, in_channels, out_channels, input_dim in lst:
    compareResults(batch_size, in_channels, out_channels, input_dim)