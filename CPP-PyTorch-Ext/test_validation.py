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

def sampleRun():
    batch_size = 1
    kernel_dim = 4
    in_channels = 4
    out_channels = 20
    input_dim = 10
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

def singleTimeCompare(batch_size, in_channels, out_channels, input_size, print_header=False,repeat_count=15):
    if print_header:
        print('{batch_size}, {in_channels}, {out_channels}, {input_size}, {durationPy}, {durationGCK}')
    kernel_size = 4
    input = torch.randn((batch_size, in_channels, input_size, input_size), requires_grad=False, dtype=torch.float32)
    kernel = torch.randn((out_channels, in_channels, kernel_size, kernel_size), requires_grad=False, dtype=torch.float32)

    # Init layers
    gckLayer = Gck4x4Layer(in_channels, out_channels, kernel_size, False, kernel)

    def func_to_measure():
        x = gckLayer.forward(input)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    print(duration)
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
    print(f'{batch_size}, {in_channels}, {out_channels}, {input_size}, {durationPy}, {durationGCK}, {faster}')

#sampleRun()
timeCompare()
#singleTimeCompare(1,1,8,1920,True)

def trial3x3():
    repeat_count = 5
    a = torch.nn.Conv2d(1,1, (3,3))
    input = torch.randn(1,1,512,512)
    
    def func_to_measure():
        x = a.forward(input)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    print(duration)
    durationWinograd = round(np.mean(duration),5)

    #Matmul clean
    result_dim = 510*510 # 512x512 after convolution
    a = torch.randn(1,9)
    #b = torch.randn(9,result_dim)
    b = torch.empty((9,510*510))
    def func_to_measure():
        
        q = 0
        for i in range(16):
            for j in range(result_dim):
                pass
        #        q += 1
        x = torch.mm(a, b)
        del x
    duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
    print(duration)
    durationMM = round(np.mean(duration),5)
    print(durationWinograd, durationMM)



#trial3x3()