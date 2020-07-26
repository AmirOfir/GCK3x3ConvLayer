#import sys
import gc
import timeit
import time
import numpy as np
import numpy.linalg as l
import torch
import torch.nn as nn
import torch.nn.functional as F
from gck_cpu_cpp import conv_fwd_3x3

repeat_count = 20



def compareTimes(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
    input = torch.randn(batch_size, in_channels, input_dim, input_dim, requires_grad=False, dtype=torch.float32).contiguous()
    linCombs = torch.randn(out_channels, in_channels * 9, requires_grad=False, dtype=torch.float32).contiguous()
    basisResultsTensor = torch.randn(in_channels*9, (input_dim-2)**2, requires_grad=False, dtype=torch.float32).contiguous()
    kernel = torch.randn(out_channels, in_channels, 3, 3,requires_grad=False, dtype=torch.float32)
    conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,3, bias=False))

    # It is running winograd no worries
    def func_to_measure():
        x = F.conv2d(input, kernel)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationWino = round(np.mean(duration),5)
    gc.collect()


    def func_to_measure():
        x = conv_fwd_3x3(input, linCombs, basisResultsTensor)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationGCK = round(np.mean(duration),5)
    gc.collect()

    del func_to_measure
    del conv
    del linCombs
    del basisResultsTensor
    del input
    gc.collect()

    return durationWino, durationGCK

print('{batch_size}, {in_channels}, {out_channels}, {input_dim}, duration Winograd , duration GCK, Winograd-GCK, GCK/Winograd')

batch_sizes_arr = [1]
in_channels_arr = [1]
out_channels_arr = [8,16,32,64,128]
input_dims = [16, 32, 64, 128, 256, 512, 650, 1024, 1280, 1500, 1920]

cross = [(a,b,c,d) for a in batch_sizes_arr for b in in_channels_arr for c in out_channels_arr for d in input_dims]
for (batch_size, in_channels, out_channels, input_dim) in cross:
    durationWino, durationGCK = compareTimes(batch_size, in_channels, out_channels, input_dim)
    diff = round(durationWino - durationGCK, 5)
    per = round((durationGCK / durationWino) * 100, 5)
    print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationWino}, {durationGCK}, {diff}, {per}%')
