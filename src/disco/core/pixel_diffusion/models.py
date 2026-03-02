import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _g(n: int, c: int):   # group norm helper
    import math as _m
    return _m.gcd(n, c) or 1

# -------------------------------------------------------------
# SinCos Positional Encoding
# -------------------------------------------------------------
class SinCosPos2D(nn.Module):
    def __init__(
        self, 
        H: int = 64, 
        W: int = 64
    ):
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

    def forward(self, B: int):
        return self.pe.unsqueeze(0).repeat(B,1,1,1)


# -------------------------------------------------------------
# Dilated Inception Residual Block
# -------------------------------------------------------------
class ResInceptionDilated(nn.Module):
    def __init__(
        self, 
        ch: int, 
        mid: int | None = None
    ):
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

    def forward(self, x: torch.Tensor):
        h = self.pre(x)
        h = self.reduce(h)
        h = torch.cat([self.b1(h), self.b2(h), self.b3(h)], dim=1)
        h = self.fuse(h)
        return x + h

# -------------------------------------------------------------
# Upsample Stage
# -------------------------------------------------------------
class UpStage(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.rb = ResInceptionDilated(ch, mid=ch//2)

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = self.conv(x)
        x = self.rb(x)
        return x

# -------------------------------------------------------------
# LatentAdapter (main module)
# -------------------------------------------------------------
class LatentAdapter(nn.Module):
    def __init__(
        self, 
        cz: int = 4, 
        cond_ch: int = 64, 
        width: int = 128,
        num_blocks_64: int = 3, 
        include_posenc: bool = True
    ):
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

    def forward(self, z64: torch.Tensor):
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Helper: Sinusoidal Timestep Embedding
# =====================================================================
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """
    Sinusoidal timestep embedding used by Diffusers.
    timesteps: [B]
    return: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if dim % 2:  # if odd, pad one extra dim
        emb = torch.cat([emb, emb[:, :1]], dim=1)
    return emb


# =====================================================================
# Basic Residual Block used in UNet
# =====================================================================
class ResidualBlock2d(nn.Module):
    """
    Simple Conv → GN → SiLU → Conv + time embedding projection
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()

        self.use_time = time_dim is not None
        if self.use_time:
            self.time_proj = nn.Linear(time_dim, out_ch)

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None):
        h = self.conv1(self.act(self.norm1(x)))

        if self.use_time:
            h = h + self.time_proj(self.act(t_emb))[:, :, None, None]

        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


# =====================================================================
# Lightweight ViT Attention on 2D maps
# =====================================================================
class ViTAttention(nn.Module):
    """
    Standard Multi-head self-attention from timm, but rewritten to avoid external import.
    """

    def __init__(
        self, dim: int, 
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0., 
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x)                 # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, C//heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimmAttn2D(nn.Module):
    """
    2D wrapper over ViT attention:
          [B, C, H, W] → flatten → MHSA → reshape
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = ViTAttention(dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.flatten(2).transpose(1, 2)   # [B, HW, C]
        h = self.attn(h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return h


# =====================================================================
# Downsample and Upsample Blocks
# =====================================================================
class Down(nn.Module):
    """Conv → Conv → AvgPool(2)"""

    def __init__(
        self, 
        in_ch: int, 
        out_ch: int,
        tdim: int,
        cond_ch: int = 0
    ):
        super().__init__()
        self.block1 = ResidualBlock2d(in_ch + cond_ch, out_ch, time_dim=tdim)
        self.block2 = ResidualBlock2d(out_ch, out_ch, time_dim=tdim)
        self.down = nn.AvgPool2d(2)

    def forward(
        self, 
        x: torch.Tensor, 
        t_emb: torch.Tensor, 
        cond: torch.Tensor | None = None
    ):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        h = self.block1(x, t_emb)
        h = self.block2(h, t_emb)
        return h, self.down(h)


class Up(nn.Module):
    """Upsample → concat skip → Conv → Conv"""

    def __init__(
        self, 
        in_ch: int,
        out_ch: int,
        tdim: int, 
        cond_ch: int = 0
    ):
        super().__init__()
        self.block1 = ResidualBlock2d(in_ch + cond_ch, out_ch, time_dim=tdim)
        self.block2 = ResidualBlock2d(out_ch, out_ch, time_dim=tdim)

    def forward(
        self, x: torch.Tensor, 
        skip: torch.Tensor, 
        t_emb: torch.Tensor, 
        cond: torch.Tensor | None = None
    ):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x


# =====================================================================
# Full UNet512 (Stage-2 Model)
# =====================================================================
class UNet512(nn.Module):
    """
    Stage-2 512×512 UNet for cascading diffusion.
    Takes:
       - noisy RGB image
       - timestep
       - multi-scale features from latent adapter
    """

    def __init__(
        self, 
        base_ch: int = 128, 
        cond_ch: int = 64, 
        time_dim: int = 256
    ):
        super().__init__()

        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        # ---------------------------
        # Timestep embedding MLP
        # ---------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(320, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # ---------------------------
        # Input: concat noisy image + s512 features
        # ---------------------------
        self.in_conv = nn.Conv2d(3 + cond_ch, ch1, 3, padding=1)

        # ---------------------------
        # Down path (512→256→128→64)
        # ---------------------------
        self.down1 = Down(ch1, ch1, tdim=time_dim, cond_ch=0)
        self.down2 = Down(ch1, ch2, tdim=time_dim, cond_ch=cond_ch)
        self.down3 = Down(ch2, ch3, tdim=time_dim, cond_ch=cond_ch)

        # ---------------------------
        # Bottleneck at 64×64
        # ---------------------------
        self.attn64 = TimmAttn2D(dim=ch3, num_heads=4)
        self.mid1 = ResidualBlock2d(ch3 + cond_ch, ch3, time_dim=time_dim)
        self.mid2 = ResidualBlock2d(ch3, ch3, time_dim=time_dim)

        # ---------------------------
        # Up path (64→128→256→512)
        # ---------------------------
        self.up3 = Up(ch3 + ch3, ch2, tdim=time_dim, cond_ch=cond_ch)
        self.up2 = Up(ch2 + ch2, ch1, tdim=time_dim, cond_ch=cond_ch)
        self.up1 = Up(ch1 + ch1, ch1, tdim=time_dim, cond_ch=0)

        # ---------------------------
        # Output layer
        # ---------------------------
        self.out_norm = nn.GroupNorm(8, ch1)
        self.out = nn.Conv2d(ch1, 3, 3, padding=1)

    # =================================================================
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        cond_feats: dict[str, torch.Tensor]
    ):
        """
        x: noisy input image        [B,3,512,512]
        timesteps:                 [B]
        cond_feats: dict {
            s64, s128, s256, s512
        }
        """

        # Timestep embedding → MLP
        t_emb = self.time_mlp(timestep_embedding(timesteps, 320))

        # -------- Input stage --------
        x = torch.cat([x, cond_feats["s512"]], dim=1)
        x = self.in_conv(x)

        # -------- Down path --------
        skip1, x = self.down1(x, t_emb, cond=None)
        skip2, x = self.down2(x, t_emb, cond_feats["s256"])
        skip3, x = self.down3(x, t_emb, cond_feats["s128"])

        # -------- Bottleneck --------
        x = self.attn64(x)
        x = torch.cat([x, cond_feats["s64"]], dim=1)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # -------- Up path --------
        x = self.up3(x, skip3, t_emb, cond_feats["s128"])
        x = self.up2(x, skip2, t_emb, cond_feats["s256"])
        x = self.up1(x, skip1, t_emb, cond=None)

        # -------- Output --------
        x = self.out(self.out_norm(x).clamp(-6, 6))
        return torch.tanh(x)
