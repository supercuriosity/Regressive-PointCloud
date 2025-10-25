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


class DynamicSTAdapter(nn.Module):
    def __init__(self, dim, rank=32, ctx_dim=128, film=True):
        super().__init__()
        self.spatial_adapt  = SpatialAdapter(dim, mix_rank=max(32, dim//4))
        self.temporal_adapt = TemporalAdapter(dim, kernel_size=3)
        self.lora_adapt     = LoRAAdapter(dim, rank=rank)

        self.ctx_enc = ContextEncoder(dim, ctx_dim=ctx_dim)

        self.gate = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

        self.film = film
        if film:
            self.to_scale = nn.Linear(ctx_dim, dim)
            self.to_shift = nn.Linear(ctx_dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, E_base, motion=None):
        B,T,N,C = E_base.shape
        ctx = self.ctx_enc(E_base, motion=motion)

        gates = F.softmax(self.gate(ctx), dim=-1)
        alpha, beta, gamma = gates[:,0], gates[:,1], gates[:,2]

        E_s = self.spatial_adapt(E_base)
        E_t = self.temporal_adapt(E_base) 

        alpha = alpha.view(B,1,1,1)
        beta  = beta.view(B,1,1,1)
        gamma = gamma.view(B,1,1,1)

        E = alpha*E_base + beta*E_s + gamma*E_t

        if self.film:
            scale = self.to_scale(ctx).view(B,1,1,C)
            shift = self.to_shift(ctx).view(B,1,1,C)
            E = E * (1 + torch.tanh(scale)) + shift

        return E, gates