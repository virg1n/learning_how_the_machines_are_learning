import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dims, n_heads):
        super().__init__()
        self.qkv = nn.Linear(dims, 3*dims, bias=False)
        self.wo = nn.Linear(dims, dims, bias=False)

        self.n_heads = n_heads
        self.head_dim = dims//n_heads


    def forward(self, x, use_causal_mask=False):
        # x = (B, T, D)
        B, T, D = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # B, n_heads, T, head_dim
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # B, n_heads, T, head_dim
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # B, n_heads, T, head_dim

        attention = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if use_causal_mask:
            mask = torch.ones_like(attention, dtype=torch.bool).triu(1) 
            attention.masked_fill_(mask, -torch.inf) 

        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)

        output = output.transpose(1, 2)
        output = output.view(B, T, D)

        return self.wo(output)



