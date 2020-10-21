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

device = torch.device('cpu')#"cuda" if torch.cuda.is_available() else "cpu")
model = Darknet('config/yolov3.cfg', use_fastconv=True).to(device)
print(model)
print('done')