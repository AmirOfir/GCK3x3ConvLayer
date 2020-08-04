#import sys
import gc
import timeit
import time
import numpy as np
import numpy.linalg as l
import torch
import torch.nn as nn
import torch.nn.functional as F
from gck_gpu_cpp import matmul_only1

repeat_count = 10

def compareTimes(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
    result_dim = input_dim - 2
    input = torch.randn(batch_size, in_channels, input_dim, input_dim, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    linCombs = torch.randn(out_channels, in_channels * 9, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    kernel = torch.randn(out_channels, in_channels, 3, 3,requires_grad=False, dtype=torch.float32).cuda()

    # Helpers
    basisResultsTensor = torch.empty(in_channels*9, (input_dim-2)**2, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    colwiseResults = torch.empty(3, input_dim * result_dim, requires_grad=False, dtype=torch.float32).cuda().contiguous()

    def func_to_measure():
        x = F.conv2d(input, kernel)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationConv = round(np.mean(duration),5)
    gc.collect()

    x = matmul_only1(input, linCombs, basisResultsTensor, colwiseResults)
    del x
    def func_to_measure():
        x = matmul_only1(input, linCombs, basisResultsTensor, colwiseResults)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    #print(duration)
    durationMatmul = round(np.mean(duration),5)
    gc.collect()
    #durationMatmul = 0

    del func_to_measure
    del linCombs
    del input
    del basisResultsTensor
    del colwiseResults
    gc.collect()

    return durationConv, durationMatmul


batch_sizes_arr = [1] #, 5,10]
in_channels_arr = [32, 64, 128, 256, 512] #[1,4,8,16, 32, 64, 128, 256, 512]
out_channels_arr = [64,128]
input_dims = [16, 32, 64, 128, 256, 512, 650, 1024]#, 1280, 1500, 1920]
print('{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationConv}, {durationMatmul}, {diff}')

cross = [(a,b,c,d) for a in batch_sizes_arr for b in in_channels_arr for c in out_channels_arr for d in input_dims]
durationConv, durationMatmul = compareTimes(1, 1, 1, 10)

for (batch_size, in_channels, out_channels, input_dim) in cross:
    durationConv, durationMatmul = compareTimes(batch_size, in_channels, out_channels, input_dim)
    diff = round(durationMatmul / durationConv, 2)
    print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationConv}, {durationMatmul},{diff}')
