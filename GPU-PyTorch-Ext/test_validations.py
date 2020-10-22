import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
import timeit
import time
from FastConv_Gpu import matmul_only, conv_fwd_3x3

kernel_dim = 3

def print_tensors_equal(a,b):
    b = torch.allclose(a, b, atol=0.01)
    if (b):
        print('same: True')
    else:
        print('Same: False (diff:', ((a-b).max()), ')')

#  LinCombs shape: (out_channels, in_channels*9). Assumed to be contiguous
#  basisResultsTensor shape: (in_channels*9, H*W). Assumed to be contiguous. values are discarded
#  colwiseResults shape (3, inputDim * resultDim). input_dim rows, result_dim cols

def test_matmul_only(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
    input = torch.randint(0,100, (batch_size, in_channels, input_dim, input_dim), dtype=torch.float32).cuda()
    linCombs = torch.randint(0,100, (out_channels, in_channels*9), requires_grad=False, dtype=torch.float32).cuda().contiguous()
    basisResultsTensor = torch.randint(0, 100, (in_channels*9, (input_dim-2)**2), requires_grad=False, dtype=torch.float32).cuda().contiguous()
    colwiseResults = torch.empty(5000, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    result = matmul_only(input, linCombs, basisResultsTensor, colwiseResults)
    print(linCombs.shape, basisResultsTensor.shape )
    expected = linCombs @ basisResultsTensor
    for i in range(batch_size):
        print_tensors_equal(result[i], expected)

def test_conv_fwd_3x3(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
#  LinCombs shape: (out_channels, in_channels*9). Assumed to be contiguous
#  basisResultsTensor shape: (in_channels*9, H*W). Assumed to be contiguous. values are discarded
#  colwiseResults shape (3, inputDim * resultDim). input_dim rows, result_dim cols
    input = torch.randint(0,100, (batch_size, in_channels, input_dim, input_dim), dtype=torch.float32).cuda()
    linCombs = torch.randint(0,100, (out_channels, in_channels*9), requires_grad=False, dtype=torch.float32).cuda().contiguous()
    basisResultsTensor = torch.randint(0, 100, (in_channels*9, (input_dim-2)**2), requires_grad=False, dtype=torch.float32).cuda().contiguous()
    colwiseResults = torch.empty(5000, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    result = matmul_only(input, linCombs, basisResultsTensor, colwiseResults)
    print(linCombs.shape, basisResultsTensor.shape )
    expected = linCombs @ basisResultsTensor
    for i in range(batch_size):
        print_tensors_equal(result[i], expected)

# def compareResults(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
#     input = torch.randint(0,100, (batch_size, in_channels, input_dim, input_dim), dtype=torch.float32)
#     kernels = torch.ones(out_channels, in_channels, kernel_dim, kernel_dim, dtype=torch.float32)
#     expected = F.conv2d(input, kernels)

#     layer = GCK3x3Layer(in_channels, out_channels, 3, False, input_dim - 2, kernels)
#     result = layer.forward(input)

#     tensors_equal(expected, result)


lst = [
    (1,1,8,1024),
    (1,1,16,256),
    (1,1,64,512),
    (1,1,128,512),
    (1,8,16,128)
    ]

for batch_size, in_channels, out_channels, input_dim in lst:
    test_conv_fwd_3x3(batch_size, in_channels, out_channels, input_dim)
    # compareResults(batch_size, in_channels, out_channels, input_dim)