import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
import timeit
import time


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

