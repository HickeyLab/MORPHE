import math
import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class PositionEncoding2D(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.register_buffer("pos_embed", self.build(num_patches, dim), persistent=False)

    def build(self, num_patches, dim):
        pe = torch.zeros(num_patches, dim)
        pos = torch.arange(num_patches).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        return pe.unsqueeze(0)  # [1, num_patches, dim]

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1)]


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c)
        )
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class CondEncoder(nn.Module):
    def __init__(self, in_channels=4, embed_dim=736, num_tokens=64):
        super().__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64),
            nn.AvgPool2d(2),
            ResidualBlock(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock(128, 256),
            nn.AvgPool2d(2),
            nn.Conv2d(256, embed_dim, 1)
        )

        self.proj = nn.Sequential(
            nn.Flatten(2),
            Transpose(-1, -2)
        )

        self.pos = PositionEncoding2D(num_tokens, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat = self.encoder(x)      # [B, 736, 8, 8]
        tokens = self.proj(feat)    # [B, 64, 736]
        tokens = self.pos(tokens)
        tokens = self.norm(tokens)
        return tokens
