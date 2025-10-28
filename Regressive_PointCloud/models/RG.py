import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

# Set up paths for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #即models文件夹
ROOT_DIR = os.path.dirname(BASE_DIR) #即Regressive_PointCloud文件夹
# keep runtime sys.path tweaks so running scripts from repo root works
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
#确保可以读取modules文件夹下的模块
from point_4d_convolution import * 

import math
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from functools import partial
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors

# import GPT blocks
from 

# 此处要import models的模块

from ipdb import set_trace as st
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RG(nn.Module):
    def __init__(self, ):
        super(RG, self).__init__()
                 
    def forward(self, inputs):
        device = inputs.get_device()
        