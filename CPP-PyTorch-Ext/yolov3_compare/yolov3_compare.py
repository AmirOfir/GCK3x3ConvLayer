from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import gc
import timeit

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gck_cpu_cpp import conv_fwd_3x3
from gck_layer import GCK3x3Layer
repeat_count = 50

regular_model = Darknet('config/yolov3.cfg', use_fastconv=False); regular_model.eval()
fastconv_model = Darknet('config/yolov3.cfg', use_fastconv=True); fastconv_model.eval()

input = torch.randn(1,3,416,416, dtype=torch.float32)

#Warmup
#regular_model(input).shape
#fastconv_model(input).shape
x = fastconv_model(input); del x
print(fastconv_model)
exit()

x = regular_model(input); del x
def func_to_measure():
	x = regular_model(input)
	del x
duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
duration = round(np.mean(duration[1:]),4)
print('YOLO_V3_Model_Paper', duration)

x = fastconv_model(input); del x
def func_to_measure():
	x = fastconv_model(input)
	del x
duration = timeit.repeat(func_to_measure, repeat=repeat_count, number=1)
duration = round(np.mean(duration[1:]),4)
gc.collect()
print('Our_Layer_Model', duration)