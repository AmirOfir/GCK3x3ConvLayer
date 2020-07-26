import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
import timeit
import time
from gck_layer import GCK3x3Layer

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

    layer = Gck4x4Layer(in_channels, out_channels, 4, False, kernels)
    
    #layer.compareBasisConv(input)

    regular_basis_conv = layer.convolveWithBasisRegular(input)

    gck_basis_conv = layer.convolveWithBasisGCK(input)

    result = layer.forward(input)
    print('regular_basis_conv', regular_basis_conv[0][0])
    print('gck_basis_conv', gck_basis_conv[0][0])
    print('result', result[0][0])
    print('expected', expected[0][0])
    tensors_equal(expected, result)
    tensors_equal(expected, regular_basis_conv)
    tensors_equal(expected, gck_basis_conv)

