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
repeat_count = 106

# Compare YOLO_Lite
input = torch.randn(1,3,224,224, dtype=torch.float32)
YOLO_Lite_Model_Paper = nn.Sequential(#nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 3, 226, 226])
						nn.Conv2d(3,16,kernel_size=3, bias=False, padding=1), #torch.Size([1, 16, 224, 224])
						nn.BatchNorm2d(16, momentum=0.9, eps=1e-5), #torch.Size([1, 16, 224, 224])
						nn.LeakyReLU(0.1), #torch.Size([1, 16, 224, 224])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 16, 112, 112])

						#n.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 16, 114, 114])
						nn.Conv2d(16,32,kernel_size=3, bias=False, padding=1), #torch.Size([1, 32, 112, 112])
						nn.BatchNorm2d(32, momentum=0.9, eps=1e-5), #torch.Size([1, 32, 112, 112])
						nn.LeakyReLU(0.1), #torch.Size([1, 32, 112, 112])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 32, 56, 56])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 32, 58, 58])
						nn.Conv2d(32, 64, kernel_size=3, bias=False, padding=1), #torch.Size([1, 64, 56, 56])
						nn.BatchNorm2d(64, momentum=0.9, eps=1e-5), #torch.Size([1, 64, 56, 56])
						nn.LeakyReLU(0.1), #torch.Size([1, 64, 56, 56])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 64, 28, 28])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 64, 30, 30])
						nn.Conv2d(64, 128, kernel_size=3, bias=False, padding=1), #torch.Size([1, 128, 28, 28])
						nn.BatchNorm2d(128, momentum=0.9, eps=1e-5), #torch.Size([1, 128, 28, 28])
						nn.LeakyReLU(0.1), #torch.Size([1, 128, 28, 28])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 128, 14, 14])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 128, 16, 16])
						nn.Conv2d(128, 128, kernel_size=3, bias=False, padding=1), #torch.Size([1, 128, 14, 14])
						nn.BatchNorm2d(128, momentum=0.9, eps=1e-5), #torch.Size([1, 128, 14, 14])
						nn.LeakyReLU(0.1), #torch.Size([1, 128, 14, 14])

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)), #torch.Size([1, 128, 7, 7])

						# nn.ZeroPad2d((1, 1, 1, 1)), #torch.Size([1, 128, 9, 9])
						nn.Conv2d(128, 256, kernel_size=3, bias=False, padding=1), #torch.Size([1, 256, 7, 7])
						nn.BatchNorm2d(256, momentum=0.9, eps=1e-5), #torch.Size([1, 256, 7, 7])
						nn.LeakyReLU(0.1), #torch.Size([1, 256, 7, 7])

						nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(256, 125, kernel_size=1, bias=False))
YOLO_Lite_Model_Paper.eval()

Our_Layer_Model = nn.Sequential(#nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(3,16,kernel_size=3, bias=False, result_dim=224),#, padding=1),
						nn.BatchNorm2d(16, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(16,32,kernel_size=3, bias=False, padding=1), #torch.Size([1, 32, 112, 112])
						nn.BatchNorm2d(32, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(32, 64, kernel_size=3, bias=False, padding=1), #torch.Size([1, 64, 56, 56])
						nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(64, 128, kernel_size=3, bias=False, result_dim=28),#, padding=1),
						nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(128, 128, kernel_size=3, bias=False, result_dim=14),#, padding=1),
						nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2)),

						# nn.ZeroPad2d((1, 1, 1, 1)),
						GCK3x3Layer(128, 256, kernel_size=3, bias=False, result_dim=7),#, padding=1),
						nn.BatchNorm2d(256, momentum=0.9, eps=1e-5),
						nn.LeakyReLU(0.1),

						nn.ZeroPad2d((1, 1, 1, 1)),
						nn.Conv2d(256, 125, kernel_size=1, bias=False))
Our_Layer_Model.eval()

x = YOLO_Lite_Model_Paper(input); del x
def func_to_measure():
	x = YOLO_Lite_Model_Paper(input)
	del x
duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
print(np.var(duration[1:]))
duration = round(np.mean(duration[1:]),4)
print('YOLO_Lite_Model_Paper', duration)

x = Our_Layer_Model(input); del x
def func_to_measure():
	x = Our_Layer_Model(input)
	del x
duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
print(np.var(duration[1:]))
duration = round(np.mean(duration[1:]),4)
gc.collect()
print('Our_Layer_Model', duration)