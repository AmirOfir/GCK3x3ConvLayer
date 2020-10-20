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
repeat_count = 100


def compareTimes(batch_size: int, in_channels: int, out_channels: int, input_dim: int, limit_lowend: bool=False):
    if limit_lowend and in_channels > out_channels:
        return 1e-6,1e-6
    input = torch.randn(batch_size, in_channels, input_dim, input_dim, requires_grad=False, dtype=torch.float32).contiguous()
    linCombs = torch.randn(out_channels, in_channels * 9, requires_grad=False, dtype=torch.float32).contiguous()
    basisResultsTensor = torch.randn(in_channels*9, (input_dim-2)**2, requires_grad=False, dtype=torch.float32).contiguous()
    kernel = torch.randn(out_channels, in_channels, 3, 3,requires_grad=False, dtype=torch.float32)
    conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,3, bias=False, padding=1))
    conv.eval()

    # It is running winograd. Double checked the code. no worries.
    def func_to_measure():
        x = conv(input) # F.conv2d(input, kernel)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationWino = round(np.mean(duration[1:]),5)
    gc.collect()

    gckLayer = GCK3x3Layer(in_channels, out_channels, result_dim=input_dim, kernel_size=3, bias=False, padding=1, kernels=kernel)

    def func_to_measure():
        x = gckLayer.forward(input)
        #x = conv_fwd_3x3(input, linCombs, basisResultsTensor)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationGCK = round(np.mean(duration[1:]),5)
    gc.collect()

    def func_to_measure():
        x = gckLayer.matmul_only(input)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    durationMatmul = round(np.mean(duration[1:]),5)
    gc.collect()

    del func_to_measure
    del conv
    del linCombs
    del basisResultsTensor
    del input
    del gckLayer
    gc.collect()

    return durationWino, durationGCK, durationMatmul

print('{batch_size}, {in_channels}, {out_channels}, {input_dim}, duration Winograd, duration matmul, duration GCK, GCK / Winograd')

cross = []
# YOLO-LITE
cross.append((1,3,16,224))
cross.append((1,16,32,112))
cross.append((1,32,64,56))
cross.append((1,64,128,28))
cross.append((1,128,128,14))
cross.append((1,128,256,7))

print('YOLO-LITE')
for (batch_size, in_channels, out_channels, input_dim) in cross:
    durationWino, durationGCK, durationMatmul = compareTimes(batch_size, in_channels, out_channels, input_dim)
    diff = round(durationWino - durationGCK, 5)
    per = round((durationGCK / durationWino) * 100, 5)

    print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationWino}, {durationMatmul}, {durationGCK}, {per}%, ')

# Tiny-YOLOV3
cross=[]
cross.append((1,3,16,418))
cross.append((1,16,32,210))
cross.append((1,32,64,106))
cross.append((1,64,128,54))
cross.append((1,128,256,28))
cross.append((1,256,512,15))
cross.append((1,512,1024,8))
cross.append((1,1024,1024,8))

print('Tiny-YOLO')
for (batch_size, in_channels, out_channels, input_dim) in cross:
    durationWino, durationGCK, durationMatmul = compareTimes(batch_size, in_channels, out_channels, input_dim)
    diff = round(durationWino - durationGCK, 5)
    per = round((durationGCK / durationWino) * 100, 5)

    print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationWino}, {durationMatmul}, {durationGCK}, {per}%, ')


# YoloV3
cross=[]
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

print('YoloV3')
for (batch_size, in_channels, out_channels, input_dim) in cross:
    durationWino, durationGCK, durationMatmul = compareTimes(batch_size, in_channels, out_channels, input_dim)
    diff = round(durationWino - durationGCK, 5)
    per = round((durationGCK / durationWino) * 100, 5)

    print(f'{batch_size}, {in_channels}, {out_channels}, {input_dim}, {durationWino}, {durationMatmul}, {durationGCK}, {per}%, ')

