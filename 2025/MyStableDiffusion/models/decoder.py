import torch
import torch.nn as nn
from attention import SelfAttention
from .encoder import VAE_Attention_Layer, VAE_Residual_Layer

class Decoder(nn.Module):
    def __init__(self, out_channels=3, in_channels=4, output_dims=128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1, padding=0),
            nn.Conv2d(in_channels, output_dims * 4, 3, padding=1),

            VAE_Residual_Layer(output_dims * 4, output_dims * 4),
            VAE_Attention_Layer(output_dims * 4),
            VAE_Residual_Layer(output_dims * 4, output_dims * 4),

            VAE_Residual_Layer(output_dims * 4, output_dims * 4),
            VAE_Residual_Layer(output_dims * 4, output_dims * 4),
            VAE_Residual_Layer(output_dims * 4, output_dims * 4),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(output_dims * 4, output_dims * 4, 3, padding=1),

            VAE_Residual_Layer(output_dims * 4, output_dims * 4),
            VAE_Residual_Layer(output_dims * 4, output_dims * 4),
            VAE_Residual_Layer(output_dims * 4, output_dims * 4),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(output_dims * 4, output_dims * 4, 3, padding=1),

            VAE_Residual_Layer(output_dims * 4, output_dims * 2), 
            VAE_Residual_Layer(output_dims * 2, output_dims * 2),
            VAE_Residual_Layer(output_dims * 2, output_dims * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(output_dims * 2, output_dims * 2, 3, padding=1),

            VAE_Residual_Layer(output_dims * 2, output_dims), 
            VAE_Residual_Layer(output_dims, output_dims),
            VAE_Residual_Layer(output_dims, output_dims),
            
            nn.GroupNorm(32, output_dims),
            nn.SiLU(),
            nn.Conv2d(output_dims, out_channels, 3, padding=1),
        ])

    def forward(self, x):
        x /= 0.18215
        for layer in self.layers:
            x = layer(x)
        return x