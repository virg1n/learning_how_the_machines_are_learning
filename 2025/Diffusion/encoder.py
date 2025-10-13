import torch
import torch.nn as nn
from attention import SelfAttention


class VAE_Residual_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),

            nn.Conv2d(in_channels, out_channels, 3, padding=1),

            nn.GroupNorm(32, out_channels),
            nn.SiLU(),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if in_channels == out_channels:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        residue = x

        x = self.main(x)
        return self.res_layer(residue) + x


class VAE_Attention_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(channels, 1)

    def forward(self, x):
        # x = (B, 512, W, H)
        residue = x
        x = self.group_norm(x)

        b, c, w, h = x.size()
        x = x.view(b, c, w * h)
        x = x.transpose(-1, -2) # (B, W * H, 512)

        x = self.attention(x)

        x = x.transpose(-1, -2).contiguous().view(b, c, w, h)
        x = x + residue
        return x



class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 128, 3, padding=1),      # (B, 128, W, H)
            VAE_Residual_Layer(128, 128),
            VAE_Residual_Layer(128, 128),
            
            nn.Conv2d(128, 128, 3, stride=2, padding=1),    # (B, 128, W/2, H/2)
            VAE_Residual_Layer(128, 256),
            VAE_Residual_Layer(256, 256),

            nn.Conv2d(256, 256, 3, stride=2, padding=1),    # (B, 256, W/4, H/4)
            VAE_Residual_Layer(256, 512),
            VAE_Residual_Layer(512, 512),

            nn.Conv2d(512, 512, 3, stride=2, padding=1),    # (B, 512, W/8, H/8)
            VAE_Residual_Layer(512, 512),
            VAE_Residual_Layer(512, 512),
            
            VAE_Residual_Layer(512, 512),                   # (B, 512, W/8, H/8)
            VAE_Attention_Layer(512),

            VAE_Residual_Layer(512, 512),                   # (B, 512, W/8, H/8)
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            nn.Conv2d(512, 8, 3, padding=1),
            nn.Conv2d(8, 8, 1, padding=0)           # (B, 8, W/8, H/8)
        ]) 
        


    def forward(self, x, noise):
        for layer in self.layers:
            x = layer(x)

        mean, log_var = x.chunk(2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)

        var = log_var.exp()
        std = var.sqrt( )

        return (mean + std * noise) * 0.18215