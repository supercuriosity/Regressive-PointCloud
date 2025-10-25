import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import sys 
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import *
from DSTAdapter import DynamicSTAdapter

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.spatial_op = nn.Sequential(
            nn.Linear(3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim_head)
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, xyzs, features):
        h = self.heads
        b, l, n, _ = features.shape
        m = l * n

        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h=h), qkv)

        xyzs_flatten = rearrange(xyzs, 'b l n d -> b (l n) d')

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attention_logits = dots

        attn = attention_logits.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        delta_xyzs = xyzs_flatten[:, None, :, :] - xyzs_flatten[:, :, None, :]

        attn = torch.unsqueeze(input=attn, dim=4)                          
        delta_full = torch.unsqueeze(input=delta_xyzs, dim=1)
        delta_full = torch.sum(input=attn*delta_full, dim=3, keepdim=False)
        spatial_bias = self.spatial_op(delta_full)

        out = out + spatial_bias
        
        out = rearrange(out, 'b h (l n) d -> b l n (h d)', l=l, n=n)
        out = self.to_out(out)
        
        return out + features

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features

class DSTA4D(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,
                 temporal_kernel_size, temporal_stride,
                 dim, depth, heads, dim_head, dropout1,
                 mlp_dim, num_classes, dropout2):
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.adapt = DynamicSTAdapter(dim=dim, rank=32, ctx_dim=128, film=True)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        
        features = features.permute(0, 1, 3, 2)

        output = self.transformer(xyzs, features)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output

