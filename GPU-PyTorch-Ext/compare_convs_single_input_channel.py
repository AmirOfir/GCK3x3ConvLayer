#import sys
import gc
import timeit
import time
import numpy as np
import numpy.linalg as l
import torch
import torch.nn as nn
import torch.nn.functional as F
from FastConv_Gpu import conv_fwd_3x3

repeat_count = 10

def compareTimes(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
    result_dim = input_dim - 2
    input = torch.randn(batch_size, in_channels, input_dim, input_dim, requires_grad=False, dtype=torch.float32).cuda().contiguous()

    kernel = torch.randn(out_channels, in_channels, 3, 3,requires_grad=False, dtype=torch.float32).cuda()
    durationConv = 0
    for _ in range(1000):
        start = time.time()
        x = F.conv2d(input, kernel)
        torch.cuda.synchronize()
        durationConv += time.time() - start
        del x
    durationConv = durationConv * 1e6/1e5
    del kernel
    gc.collect()

    # Helpers
    basisResultsTensor = torch.empty(in_channels*9, (input_dim-2)**2, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    colwiseResults = torch.empty(3, input_dim * input_dim, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    linCombs = torch.randn(out_channels, in_channels * 9, requires_grad=False, dtype=torch.float32).cuda().contiguous()
    
    # Warmup
    x = conv_fwd_3x3(input, linCombs, basisResultsTensor, colwiseResults); torch.cuda.synchronize(); del x
    
    durationMatmul = 0
    for _ in range(1000):
        start = time.time()
        x = conv_fwd_3x3(input, linCombs, basisResultsTensor, colwiseResults)
        torch.cuda.synchronize()
        durationMatmul += time.time() - start
        del x
    durationMatmul = durationMatmul * 1e6/1e5
    
    del linCombs
    del input
    del basisResultsTensor
    del colwiseResults
    gc.collect()
    torch.cuda.empty_cache()

    return durationConv, durationMatmul


batch_sizes_arr = [1]
in_channels_arr = [1] #[1,4,8,16, 32, 64, 128, 256, 512]
out_channels_arr = [1,3,9,16,32,64]
input_dims = [16, 32, 64, 128, 256, 512, 650, 1024, 1280, 1500]
print('{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationConv}, {durationMatmul}, {diff}')

cross = [(a,b,c,d) for a in batch_sizes_arr for b in in_channels_arr for c in out_channels_arr for d in input_dims]
durationConv, durationMatmul = compareTimes(1, 1, 1, 10)

meanDiffI = 0; meanDiff = 0
for (batch_size, in_channels, out_channels, input_dim) in cross:
    try:
        durationConv, durationMatmul = compareTimes(batch_size, in_channels, out_channels, input_dim)
        diff = round(durationMatmul / durationConv, 2)
        print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationConv}, {durationMatmul},{diff}')
        meanDiff += diff
        meanDiffI += 1
    finally:
        pass
    
print(meanDiff / meanDiffI)