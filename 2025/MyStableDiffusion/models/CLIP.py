import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from attention import SelfAttention


class CLIP_Layer(nn.Module):
    def __init__(self, n_heads, n_dims, n_hidden_dims = None):
        super().__init__()
        if n_hidden_dims is None: n_hidden_dims = 4 * n_dims

        self.norm_1 = nn.LayerNorm(n_dims)
        self.attention = SelfAttention(n_dims, n_heads)

        self.norm_2 = nn.LayerNorm(n_dims)
        self.up = nn.Linear(n_dims, n_hidden_dims)
        self.down = nn.Linear(n_hidden_dims, n_dims)

    def forward(self, x):
        residue = x

        x = self.norm_1(x)
        x = self.attention(x, use_causal_mask=True)

        x = x + residue
        residue = x

        x = self.norm_2(x)
        x = self.up(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU
        x = self.down(x)

        return x + residue

        


class CLIP(nn.Module):
    def __init__(self, vocab_size, n_dims, n_tokens, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_dims)
        self.pos_embedding = nn.Parameter(torch.zeros(n_tokens, n_dims))

        self.layers = nn.Sequential(*[CLIP_Layer(n_heads, n_dims) for _ in range(12)])

        self.layernorm = nn.LayerNorm(n_dims)

    def forward(self, x):
        embedded = self.embedding(x)
        x = embedded + self.pos_embedding

        x = self.layers(x)

        return self.layernorm(x)
