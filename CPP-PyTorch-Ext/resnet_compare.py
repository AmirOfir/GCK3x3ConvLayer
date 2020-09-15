#import sys
import gc
import timeit
import time
import numpy as np
import numpy.linalg as l
import torch
import torch.nn as nn
import torch.nn.functional as F
from gck_layer import GCK3x3Layer
repeat_count = 20
from resnet_nets import r
