import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, dims, out_dims=None):
        super().__init__()
        if out_dims is None:
            out_dims = dims * 4
        self.up = nn.Linear(dims, out_dims)
        self.out = nn.Linear(out_dims, out_dims)
    
    def forward(self, t):
        t = self.up(t)
        t = F.silu(t)
        return self.out(t)


class MySequential(nn.Sequential):
    def forward(self, x, prompt, time):
        for layer in self:
            if isinstance(layer, UNET_Attention):
                x = layer(x, prompt)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET_Attention(nn.Module):
    def __init__(self, n_heads, head_dims, dim_prompt):
        super().__init__()
        channels = n_heads * head_dims

        self.group_norm = nn.GroupNorm(32, channels)
        self.conv_in = nn.Conv2d(channels, channels, 1)

        self.layer_norm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(channels, n_heads, in_proj_bias=False)

        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(channels, n_heads, dim_prompt, in_proj_bias=False)

        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, prompt):
        long_residue = x
        x = self.group_norm(x)
        x = self.conv_in(x)

        b, c, w, h = x.shape
        x = x.view(b, c, w * h)
        x = x.transpose(-1, -2) # (B, W * H, 512)

        short_residue = x
        x = self.layer_norm_1(x)
        x = self.attention_1(x)

        x = x + short_residue

        short_residue = x
        x = self.layer_norm_2(x)
        x = self.attention_2(x, prompt)

        x = x + short_residue

        short_residue = x
        x = self.layer_norm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x = x + short_residue

        x = x.transpose(-1, -2).contiguous().view(b, c, w, h)

        return self.conv_out(x) + long_residue




class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_time):
        super().__init__()
        self.group_norm_first = nn.GroupNorm(32, in_channels)
        self.conv_first = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # self.first = nn.Sequential(
        #     nn.GroupNorm(32, in_channels),
        #     # nn.SiLU(),s

        #     nn.Conv2d(in_channels, out_channels, 3, padding=1),
        # )

        self.linear_time = nn.Linear(dim_time, out_channels)

        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # self.merged = nn.Sequential(
        #     nn.GroupNorm(32, out_channels),
        #     nn.SiLU(),

        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # )

        if in_channels == out_channels:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x, time):
        residue = x

        x = self.group_norm_first(x)
        x = F.silu(x)
        x = self.conv_first(x)

        time = F.silu(time)
        time = self.linear_time(time)

        x = x+ time.unsqueeze(-1).unsqueeze(-1)
        
        x = self.group_norm_merged(x)
        x = F.silu(x)
        x = self.conv_merged(x)

        return self.res_layer(residue) + x
    

class UNET(nn.Module):
    def __init__(self, in_channels=4, hiddens_channels=320, attn_emb=40, dim_time=1280, dim_prompt=768):
        super().__init__()
        self.encoders = nn.ModuleList([
            MySequential(nn.Conv2d(in_channels, hiddens_channels, 3, padding=1)),
            MySequential(UNET_ResidualBlock(hiddens_channels, hiddens_channels, dim_time), UNET_Attention(8, attn_emb, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels, hiddens_channels, dim_time), UNET_Attention(8, attn_emb, dim_prompt)),

            MySequential(nn.Conv2d(hiddens_channels, hiddens_channels, 3, stride=2, padding=1)),
            MySequential(UNET_ResidualBlock(hiddens_channels, hiddens_channels*2, dim_time), UNET_Attention(8, attn_emb*2, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*2, hiddens_channels*2, dim_time), UNET_Attention(8, attn_emb*2, dim_prompt)),

            MySequential(nn.Conv2d(hiddens_channels*2, hiddens_channels*2, 3, stride=2, padding=1)),
            MySequential(UNET_ResidualBlock(hiddens_channels*2, hiddens_channels*4, dim_time), UNET_Attention(8, attn_emb*4, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*4, hiddens_channels*4, dim_time), UNET_Attention(8, attn_emb*4, dim_prompt)),

            MySequential(nn.Conv2d(hiddens_channels*4, hiddens_channels*4, 3, stride=2, padding=1)),
            MySequential(UNET_ResidualBlock(hiddens_channels*4, hiddens_channels*4, dim_time)),
            MySequential(UNET_ResidualBlock(hiddens_channels*4, hiddens_channels*4, dim_time)),
        ])

        self.bottom = MySequential(
            UNET_ResidualBlock(hiddens_channels*4, hiddens_channels*4, dim_time),
            UNET_Attention(8, attn_emb*4, dim_prompt),
            UNET_ResidualBlock(hiddens_channels*4, hiddens_channels*4, dim_time)
        )

        self.decoders = nn.ModuleList([
            MySequential(UNET_ResidualBlock(hiddens_channels*8, hiddens_channels*4, dim_time)),
            MySequential(UNET_ResidualBlock(hiddens_channels*8, hiddens_channels*4, dim_time)),
            MySequential(UNET_ResidualBlock(hiddens_channels*8, hiddens_channels*4, dim_time), nn.Conv2d(hiddens_channels*4, hiddens_channels*4, kernel_size=3, padding=1), nn.Upsample(scale_factor=2)),

            MySequential(UNET_ResidualBlock(hiddens_channels*8, hiddens_channels*4, dim_time), UNET_Attention(8, attn_emb*4, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*8, hiddens_channels*4, dim_time), UNET_Attention(8, attn_emb*4, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*6, hiddens_channels*4, dim_time), UNET_Attention(8, attn_emb*4, dim_prompt), nn.Conv2d(hiddens_channels*4, hiddens_channels*4, kernel_size=3, padding=1), nn.Upsample(scale_factor=2, mode='nearest')),

            MySequential(UNET_ResidualBlock(hiddens_channels*6, hiddens_channels*2, dim_time), UNET_Attention(8, attn_emb*2, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*4, hiddens_channels*2, dim_time), UNET_Attention(8, attn_emb*2, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*3, hiddens_channels*2, dim_time), UNET_Attention(8, attn_emb*2, dim_prompt), nn.Conv2d(hiddens_channels*2, hiddens_channels*2, kernel_size=3, padding=1), nn.Upsample(scale_factor=2, mode='nearest')),

            MySequential(UNET_ResidualBlock(hiddens_channels*3, hiddens_channels*1, dim_time), UNET_Attention(8, attn_emb, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*2, hiddens_channels*1, dim_time), UNET_Attention(8, attn_emb, dim_prompt)),
            MySequential(UNET_ResidualBlock(hiddens_channels*2, hiddens_channels*1, dim_time), UNET_Attention(8, attn_emb, dim_prompt)),
        ])

    def forward(self, x, prompt, time):

        residues = []
        for layer in self.encoders:
            x = layer(x, prompt, time)
            residues.append(x)

        x = self.bottom(x, prompt, time)

        for layer in self.decoders:
            x = torch.cat((x, residues.pop()), dim = 1)
            x = layer(x, prompt, time)
        
        return x


class UNET_Final_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = F.silu(x)
        return self.conv(x)



class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_Final_Layer(320, 4)

    def forward(self, z, prompt, time):
        time = self.time_embedding(time)

        z = self.unet(z, prompt, time)

        return self.final(z)