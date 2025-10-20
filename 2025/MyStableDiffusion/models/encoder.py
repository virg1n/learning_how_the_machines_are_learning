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
        # x = (B, input_dims * 4, W, H)
        residue = x
        x = self.group_norm(x)

        b, c, w, h = x.shape
        x = x.view(b, c, w * h)
        x = x.transpose(-1, -2) # (B, W * H, input_dims * 4)

        x = self.attention(x)

        x = x.transpose(-1, -2).contiguous().view(b, c, w, h)
        x = x + residue
        return x



class Encoder(nn.Module):
    def __init__(self, in_channels=3, input_dims=128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 128, 3, padding=1),      # (B, 128, W, H)
            VAE_Residual_Layer(input_dims, input_dims),
            VAE_Residual_Layer(input_dims, input_dims),
            
            nn.Conv2d(input_dims, input_dims, 3, stride=2, padding=1),    # (B, 128, W/2, H/2)
            VAE_Residual_Layer(input_dims, input_dims * 2),
            VAE_Residual_Layer(input_dims * 2, input_dims * 2),

            nn.Conv2d(input_dims * 2, input_dims * 2, 3, stride=2, padding=1),    # (B, 128 * 2, W/4, H/4)
            VAE_Residual_Layer(input_dims * 2, input_dims * 4),
            VAE_Residual_Layer(input_dims * 4, input_dims * 4),

            nn.Conv2d(input_dims * 4, input_dims * 4, 3, stride=2, padding=1),    # (B, 128 * 4, W/8, H/8)
            VAE_Residual_Layer(input_dims * 4, input_dims * 4),
            VAE_Residual_Layer(input_dims * 4, input_dims * 4),
            
            VAE_Residual_Layer(input_dims * 4, input_dims * 4),                   # (B, 128 * 4, W/8, H/8)
            VAE_Attention_Layer(input_dims * 4),

            VAE_Residual_Layer(input_dims * 4, input_dims * 4),                   # (B, 128 * 4, W/8, H/8)
            nn.GroupNorm(32, input_dims * 4),
            nn.SiLU(),

            nn.Conv2d(input_dims * 4, 8, 3, padding=1),
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