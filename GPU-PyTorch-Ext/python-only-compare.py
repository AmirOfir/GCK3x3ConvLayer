import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
import timeit
import time

repeat_count = 20
device = 'cuda:0'
def compareTimes(batch_size: int, in_channels: int, out_channels: int, input_dim: int):
    input = torch.randn(batch_size, in_channels, input_dim, input_dim, requires_grad=False, dtype=torch.float32).contiguous().device(device)
    linCombs = torch.randn(out_channels, in_channels * 9, requires_grad=False, dtype=torch.float32).contiguous().device(device)
    basisResultsTensor = torch.randn(in_channels*9, (input_dim-2)**2, requires_grad=False, dtype=torch.float32).contiguous().device(device)
    kernel = torch.randn(out_channels, in_channels, 3, 3,requires_grad=False, dtype=torch.float32).device(device)
    conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,3, bias=False))

    def func_to_measure():
        x = F.conv2d(input, kernel)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationConv = round(np.mean(duration),5)
    gc.collect()

    result_dim = (input_dim-2)**2 # 512x512 after convolution
    a = torch.randn(out_channels,9)
    b = torch.empty((9,result_dim))
    def func_to_measure():
        x = torch.mm(a, b)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationMatmul = round(np.mean(duration),5)
    gc.collect()

    del func_to_measure
    del conv
    del linCombs
    del basisResultsTensor
    del input
    gc.collect()

    return durationConv, durationMatmul


batch_sizes_arr = [1]
in_channels_arr = [1,4,8,16,32]
out_channels_arr = [4, 8,16,32,64,128]
input_dims = [16, 32, 64, 128, 256, 512, 650, 1024, 1280, 1500, 1920]
print('{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationConv}, {durationMatmul}, {diff}')

cross = [(a,b,c,d) for a in batch_sizes_arr for b in in_channels_arr for c in out_channels_arr for d in input_dims]

# YOLO-LITE
cross.append((1,3,16,224))
cross.append((1,16,32,112))
cross.append((1,32,64,56))
cross.append((1,64,128,28))
cross.append((1,128,128,14))
cross.append((1,128,256,7))

# YoloV3
cross.append((1,3,32,416))
cross.append((1,32,64,208))
cross.append((1,64,32,104))
cross.append((1,32,64,52))
cross.append((1,64,128,26))
cross.append((1,128,128,13))
cross.append((1,128,256,13))
cross.append((1,256,512,13))
cross.append((1,256,512,7))
cross.append((1,512,512,7))
cross.append((1,512,1024,7))
for (batch_size, in_channels, out_channels, input_dim) in cross:
    durationConv, durationMatmul = compareTimes(batch_size, in_channels, out_channels, input_dim)
    diff = round(durationConv - durationMatmul, 5)
    per = round((durationMatmul / durationConv) * 100, 5)
    print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationConv}, {durationMatmul}, {diff}, {per}%')


