import torch
import torch.nn as nn
from attention import SelfAttention
from encoder import VAE_Attention_Layer, VAE_Residual_Layer

class Decoder(nn.Module):
    def __init__(self, out_channels=3, in_channels=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 1, padding=0),
            nn.Conv2d(in_channels, 512, 3, padding=1),

            VAE_Residual_Layer(512, 512),
            VAE_Attention_Layer(512),
            VAE_Residual_Layer(512, 512),
            VAE_Residual_Layer(512, 512),
            VAE_Residual_Layer(512, 512),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            VAE_Residual_Layer(512, 512),
            VAE_Residual_Layer(512, 512),
            VAE_Residual_Layer(512, 512),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            VAE_Residual_Layer(512, 256),
            VAE_Residual_Layer(256, 256),
            VAE_Residual_Layer(256, 256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1),
            VAE_Residual_Layer(256, 128),
            VAE_Residual_Layer(128, 128),
            VAE_Residual_Layer(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.Tanh()
        ])


    def forward(self, x):
        x /= 0.18215

        for layer in self.layers:
            x = layer(x)

        return x