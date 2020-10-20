import sys
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

# Compare Tiny_YOLO
input = torch.randn(1,3,416,416, requires_grad=False, dtype=torch.float32)

Tiny_YOLO_Model_Paper = nn.Sequential(#nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 3, 418, 418])
						nn.Conv2d(3,16,kernel_size=3, bias=False, padding=1), #torch.Size([1, 16, 416, 416])
						nn.BatchNorm2d(16, momentum=0.9, eps=1e-5), #torch.Size([1, 16, 416, 416])
						nn.LeakyReLU(0.1), #torch.Size([1, 16, 416, 416])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 16, 208, 208])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 16, 210, 210])
						nn.Conv2d(16,32,kernel_size=3, bias=False, padding=1), #torch.Size([1, 32, 208, 208])
						nn.BatchNorm2d(32, momentum=0.9, eps=1e-5), #torch.Size([1, 32, 208, 208])
						nn.LeakyReLU(0.1), #torch.Size([1, 32, 208, 208])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 32, 104, 104])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 32, 106, 106])
						nn.Conv2d(32, 64, kernel_size=3, bias=False, padding=1), #torch.Size([1, 64, 104, 104])
						nn.BatchNorm2d(64, momentum=0.9, eps=1e-5), #torch.Size([1, 64, 104, 104])
						nn.LeakyReLU(0.1), #torch.Size([1, 64, 104, 104])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 64, 52, 52])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 64, 54, 54])
						nn.Conv2d(64, 128, kernel_size=3, bias=False, padding=1), #torch.Size([1, 128, 52, 52])
						nn.BatchNorm2d(128, momentum=0.9, eps=1e-5), #torch.Size([1, 128, 52, 52])
						nn.LeakyReLU(0.1), #torch.Size([1, 128, 52, 52])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 128, 26, 26])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 128, 28, 28])
						nn.Conv2d(128, 256, kernel_size=3, bias=False, padding=1), #torch.Size([1, 256, 26, 26])
						nn.BatchNorm2d(256, momentum=0.9, eps=1e-5), #torch.Size([1, 256, 26, 26])
						nn.LeakyReLU(0.1), #torch.Size([1, 256, 26, 26])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 256, 13, 13])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 256, 15, 15])
						nn.Conv2d(256, 512, kernel_size=3, bias=False, padding=1), #torch.Size([1, 512, 13, 13])
						nn.BatchNorm2d(512, momentum=0.9, eps=1e-5), #torch.Size([1, 512, 13, 13])
						nn.LeakyReLU(0.1), #torch.Size([1, 512, 13, 13])

                        nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 512, 6, 6])

                        # nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 512, 8, 8])
						nn.Conv2d(512, 1024, kernel_size=3, bias=False, padding=1), #torch.Size([1, 1024, 6, 6])
						nn.BatchNorm2d(1024, momentum=0.9, eps=1e-5), #torch.Size([1, 1024, 6, 6])
						nn.LeakyReLU(0.1), #torch.Size([1, 1024, 6, 6])

                        # nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 1024, 8, 8])
						nn.Conv2d(1024, 1024, kernel_size=3, bias=False, padding=1), #torch.Size([1, 1024, 6, 6])
						nn.BatchNorm2d(1024, momentum=0.9, eps=1e-5), #torch.Size([1, 1024, 6, 6])
						nn.LeakyReLU(0.1), #torch.Size([1, 1024, 6, 6])

						nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(1024, 125, kernel_size=1, bias=False))
Tiny_YOLO_Model_Paper.eval()

Our_Layer_Model = nn.Sequential(#nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(3,16,kernel_size=3, bias=False,result_dim=416, padding=1),
						nn.BatchNorm2d(16, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(16,32,kernel_size=3, bias=False, padding=1),
						nn.BatchNorm2d(32, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(32, 64, kernel_size=3, bias=False, padding=1),
						nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(64, 128, kernel_size=3, bias=False, padding=1),
						nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						#GCK3x3Layer(128, 256, kernel_size=3, bias=False,),
						GCK3x3Layer(128, 256, kernel_size=3, bias=False, result_dim=26, padding=1),
						nn.BatchNorm2d(256, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(256, 512, kernel_size=3, bias=False, result_dim=13, padding=1),
						nn.BatchNorm2d(512, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

                        nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

                        # nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(512, 1024, kernel_size=3, bias=False, result_dim=6, padding=1),
						nn.BatchNorm2d(1024, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

                        # nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(1024, 1024, kernel_size=3, bias=False, result_dim=6, padding=1),
						nn.BatchNorm2d(1024, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(1024, 125, kernel_size=1, bias=False))
Our_Layer_Model.eval()

def func_to_measure():
	Our_Layer_Model(input)
duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
duration = round(np.mean(duration[1:]),4)
gc.collect()
print('Our_Layer_Model', duration)

def func_to_measure():
	Tiny_YOLO_Model_Paper(input)
duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
duration = round(np.mean(duration[1:]),4)
gc.collect()
print('Tiny_YOLO_Model_Paper', duration)

