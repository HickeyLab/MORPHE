import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transpose(nn.Module):
    def __init__(
        self, 
        dim0: int, 
        dim1: int
    ):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(
        self, 
        x: torch.Tensor
    ):
        return x.transpose(self.dim0, self.dim1)

class PositionalEncoding2D(nn.Module):
    def __init__(
        self, 
        num_patches: int, 
        dim: int
    ):
        super().__init__()
        self.register_buffer('pos_embed', self.build_sincos_encoding(num_patches, dim), persistent=False)

    def build_sincos_encoding(
        self, 
        num_patches: int, 
        dim: int
    ):
        pe = torch.zeros(num_patches, dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, num_patches, dim]

    def forward(self, x: torch.Tensor):
        return x + self.pos_embed[:, :x.size(1), :]

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.block(x) + self.skip(x)

class CondEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int = 4, 
        out_channels: int = 736, 
        num_tokens: int = 64
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64), # [B, 64, 64, 64]
            nn.AvgPool2d(2), # [B, 64, 32, 32]
            ResidualBlock(64, 128),
            nn.AvgPool2d(2), # [B, 128, 16, 16]
            ResidualBlock(128, 256),
            nn.AvgPool2d(2), # [B, 256, 8, 8]
            nn.Conv2d(256, out_channels, kernel_size=1) # [B, 736, 8, 8]
        )
        self.proj = nn.Sequential(
            nn.Flatten(2),  # [B, 736, 64]
            Transpose(-1, -2),   # [B, 64, 736]
        )
        self.pos_embed = PositionalEncoding2D(num_patches=num_tokens, dim=out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)          # [B, 736, 8, 8]
        tokens = self.proj(feat)        # [B, 64, 736]
        tokens = self.pos_embed(tokens) # [B, 64, 736]
        tokens = self.norm(tokens)
        return tokens

class CoordEncoder(nn.Module):
    def __init__(
        self, 
        embed_dim: int = 32, 
        num_tokens: int = 64
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(64 * 64, 2048),
            nn.GELU(),
            nn.Linear(2048, num_tokens * embed_dim)  # 64 * 32
        )

    def forward(
        self, 
        mask: torch.Tensor
    ):
        mask_ds = F.interpolate(mask, size=(64, 64), mode="nearest")  # (B,1,64,64)
        B = mask_ds.shape[0]
        x = mask_ds.view(B, -1)  # (B, 4096)
        x = self.mlp(x)  # (B, 64*32)
        x = x.view(B, self.num_tokens, self.embed_dim)  # (B,64,32)
        return x

class CondEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4, 
        out_channels: int = 768, 
        num_tokens: int = 64
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64), # [B, 64, 64, 64]
            nn.AvgPool2d(2), # [B, 64, 32, 32]
            ResidualBlock(64, 128),
            nn.AvgPool2d(2), # [B, 128, 16, 16]
            ResidualBlock(128, 256),
            nn.AvgPool2d(2), # [B, 256, 8, 8]
            nn.Conv2d(256, out_channels, kernel_size=1) # [B, 736, 8, 8]
        )
        self.proj = nn.Sequential(
            nn.Flatten(2),  # [B, 736, 64]
            Transpose(-1, -2),   # [B, 64, 736]
        )
        self.pos_embed = PositionalEncoding2D(num_patches=num_tokens, dim=out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)          # [B, 736, 8, 8]
        tokens = self.proj(feat)        # [B, 64, 736]
        tokens = self.pos_embed(tokens) # [B, 64, 736]
        tokens = self.norm(tokens)
        return tokens