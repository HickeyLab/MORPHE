import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _g(n, c):   # group norm helper
    import math as _m
    return _m.gcd(n, c) or 1

# -------------------------------------------------------------
# SinCos Positional Encoding
# -------------------------------------------------------------
class SinCosPos2D(nn.Module):
    def __init__(self, H=64, W=64):
        super().__init__()
        yy, xx = torch.meshgrid(
            torch.linspace(0,1,H), torch.linspace(0,1,W),
            indexing="ij"
        )
        pe = torch.stack([
            torch.sin(2*math.pi*xx), torch.cos(2*math.pi*xx),
            torch.sin(2*math.pi*yy), torch.cos(2*math.pi*yy)
        ], dim=0)
        self.register_buffer("pe", pe.float(), persistent=False)

    def forward(self, B):
        return self.pe.unsqueeze(0).repeat(B,1,1,1)


# -------------------------------------------------------------
# Dilated Inception Residual Block
# -------------------------------------------------------------
class ResInceptionDilated(nn.Module):
    def __init__(self, ch, mid=None):
        super().__init__()
        mid = mid or ch // 2

        self.pre = nn.Sequential(nn.GroupNorm(_g(8,ch), ch), nn.SiLU())
        self.reduce = nn.Conv2d(ch, mid, 1)

        self.b1 = nn.Conv2d(mid, mid, 3, padding=1, dilation=1)
        self.b2 = nn.Conv2d(mid, mid, 3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(mid, mid, 3, padding=4, dilation=4)

        self.fuse = nn.Sequential(
            nn.GroupNorm(_g(8, mid*3), mid*3),
            nn.SiLU(),
            nn.Conv2d(mid*3, ch, 1)
        )

    def forward(self, x):
        h = self.pre(x)
        h = self.reduce(h)
        h = torch.cat([self.b1(h), self.b2(h), self.b3(h)], dim=1)
        h = self.fuse(h)
        return x + h

# -------------------------------------------------------------
# Upsample Stage
# -------------------------------------------------------------
class UpStage(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.rb = ResInceptionDilated(ch, mid=ch//2)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.rb(x)
        return x

# -------------------------------------------------------------
# LatentAdapter (main module)
# -------------------------------------------------------------
class LatentAdapter(nn.Module):
    def __init__(self, cz=4, cond_ch=64, width=128,
                 num_blocks_64=3, include_posenc=True):
        super().__init__()
        self.include_posenc = include_posenc
        in_ch = cz + 4  # latent + posenc

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GroupNorm(_g(8,width), width),
            nn.SiLU()
        )

        self.blocks64 = nn.Sequential(
            *[ResInceptionDilated(width, mid=width//2)
              for _ in range(num_blocks_64)]
        )

        self.up1 = UpStage(width)
        self.up2 = UpStage(width)
        self.up3 = UpStage(width)

        def head():
            return nn.Sequential(
                nn.Conv2d(width, cond_ch, 1),
                nn.GroupNorm(_g(8,cond_ch), cond_ch),
                nn.SiLU()
            )

        self.out64  = head()
        self.out128 = head()
        self.out256 = head()
        self.out512 = head()

        self.posenc = SinCosPos2D(64,64) if include_posenc else None

    def forward(self, z64):
        B = z64.size(0)
        feats = [z64]
        if self.include_posenc:
            feats.append(self.posenc(B).to(z64.dtype).to(z64.device))

        x = torch.cat(feats, dim=1)
        x = self.in_conv(x)
        x = self.blocks64(x)

        f64 = self.out64(x)

        x = self.up1(x); f128 = self.out128(x)
        x = self.up2(x); f256 = self.out256(x)
        x = self.up3(x); f512 = self.out512(x)

        return {"s64":f64, "s128":f128, "s256":f256, "s512":f512}
