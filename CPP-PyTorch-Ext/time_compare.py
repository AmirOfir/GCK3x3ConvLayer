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
from gck_layer import GCK3x3Layer

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

    gckLayer = GCK3x3Layer(in_channels, out_channels, 3, False, input_dim - 2, kernel)

    def func_to_measure():
        x = gckLayer.forward(input)
        #x = conv_fwd_3x3(input, linCombs, basisResultsTensor)
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


lst = [
    (1,1,8,1024),
    (1,1,16,256),
    (1,1,64,512),
    (1,1,128,512),
    (1,8,16,128)
    ]

print('{batch_size}, {in_channels}, {out_channels}, {input_dim}, duration Winograd , duration GCK, Winograd-GCK')

for batch_size, in_channels, out_channels, input_dim in lst:
    durationWino, durationGCK = compareTimes(batch_size, in_channels, out_channels, input_dim)
    diff = durationWino - durationGCK
    print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationWino}, {durationGCK},{diff}')


















def timeCompare():
    batch_sizes_arr = [1,5,10]
    in_channels_arr = [1,4,8,16, 32, 64, 128, 256, 512]
    out_channels_arr = [8,16,32,64,128,256,512,1024] #[5+5*i for i in range(140)]#
    input_sizes = [16, 32, 64, 128, 256, 512, 650,  750, 1024, 1280, 1500, 1920]
    kernel_size = 3
    
    repeat_count = 10
    print('{batch_size}, {in_channels}, {out_channels}, {input_size}, {durationPy}, {durationGCK}')
    cross = [(a,b,c,d) for a in batch_sizes_arr for b in in_channels_arr for c in out_channels_arr for d in input_sizes]
    for (batch_size, in_channels, out_channels, input_size) in cross:
        input = torch.randn((batch_size, in_channels, input_size, input_size), requires_grad=False, dtype=torch.float32)
        kernel = torch.randn((out_channels, in_channels, kernel_size, kernel_size), requires_grad=False, dtype=torch.float32)

        # Init layers
        gckLayer = GCK3x3Layer(in_channels, out_channels, kernel_size, False,input_size - 2, kernel)

        def func_to_measure():
            x = gckLayer.forward(input)
            del x
        duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
        durationGCK = round(np.mean(duration),5)
        gc.collect()

        def func_to_measure():
            x = F.conv2d(input, kernel)
            del x
        duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
        durationPy = round(np.mean(duration), 5)
        gc.collect()

        time.sleep(0.2)
        faster = 'faster' if durationGCK < durationPy else 'slower'
        d = durationGCK / durationPy
        print(f'{batch_size}, {in_channels}, {out_channels}, {input_size}, {durationPy}, {durationGCK}, {faster}, {d}')
