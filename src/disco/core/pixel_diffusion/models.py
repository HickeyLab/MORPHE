import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _g(target_groups: int, num_channels: int) -> int:
    """
    Return a valid GroupNorm group count that cleanly divides `num_channels`.

    Using gcd lets us request a preferred number of groups (for example 8)
    while still guaranteeing GroupNorm receives a valid divisor.
    """
    return math.gcd(target_groups, num_channels) or 1


class SinCosPos2D(nn.Module):
    """
    Fixed 2D sinusoidal positional encoding over a spatial grid.

    Produces 4 channels:
      - sin(x), cos(x)
      - sin(y), cos(y)

    The encoding is registered as a non-persistent buffer because it is
    deterministic and can be recreated from H/W rather than stored in checkpoints.
    """

    def __init__(self, H: int = 64, W: int = 64):
        super().__init__()

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing="ij",
        )

        pe = torch.stack(
            [
                torch.sin(2 * math.pi * xx),
                torch.cos(2 * math.pi * xx),
                torch.sin(2 * math.pi * yy),
                torch.cos(2 * math.pi * yy),
            ],
            dim=0,
        )
        self.register_buffer("pe", pe.float(), persistent=False)

    def forward(self, B: int) -> torch.Tensor:
        # Expand the fixed encoding across the batch dimension.
        return self.pe.unsqueeze(0).repeat(B, 1, 1, 1)


class ResInceptionDilated(nn.Module):
    """
    Residual block with parallel dilated 3x3 convolutions.

    The three branches let the block aggregate local and larger-context
    spatial information without changing resolution.
    """

    def __init__(self, ch: int, mid: int | None = None):
        super().__init__()
        mid = mid or ch // 2

        self.pre = nn.Sequential(
            nn.GroupNorm(_g(8, ch), ch),
            nn.SiLU(),
        )
        self.reduce = nn.Conv2d(ch, mid, 1)

        self.b1 = nn.Conv2d(mid, mid, 3, padding=1, dilation=1)
        self.b2 = nn.Conv2d(mid, mid, 3, padding=2, dilation=2)
        self.b3 = nn.Conv2d(mid, mid, 3, padding=4, dilation=4)

        self.fuse = nn.Sequential(
            nn.GroupNorm(_g(8, mid * 3), mid * 3),
            nn.SiLU(),
            nn.Conv2d(mid * 3, ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.pre(x)
        h = self.reduce(h)
        h = torch.cat([self.b1(h), self.b2(h), self.b3(h)], dim=1)
        h = self.fuse(h)

        return residual + h


class UpStage(nn.Module):
    """
    Resolution-doubling stage used by the latent adapter.

    Bilinear upsampling is followed by convolution and a residual refinement block.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.rb = ResInceptionDilated(ch, mid=ch // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.rb(x)
        return x


class LatentAdapter(nn.Module):
    """
    Convert a low-resolution latent map into multi-scale conditioning features
    for the stage-2 UNet.

    Expected input:
      z64: [B, cz, 64, 64]

    Returns a dict of features aligned with the UNet resolutions:
      s64, s128, s256, s512
    """

    def __init__(
        self,
        cz: int = 4,
        cond_ch: int = 64,
        width: int = 128,
        num_blocks_64: int = 3,
        include_posenc: bool = True,
    ):
        super().__init__()
        self.include_posenc = include_posenc

        # Input is latent channels plus 4 positional-encoding channels when enabled.
        in_ch = cz + 4 if include_posenc else cz

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GroupNorm(_g(8, width), width),
            nn.SiLU(),
        )

        self.blocks64 = nn.Sequential(
            *[
                ResInceptionDilated(width, mid=width // 2)
                for _ in range(num_blocks_64)
            ]
        )

        self.up1 = UpStage(width)
        self.up2 = UpStage(width)
        self.up3 = UpStage(width)

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(width, cond_ch, 1),
                nn.GroupNorm(_g(8, cond_ch), cond_ch),
                nn.SiLU(),
            )

        self.out64 = head()
        self.out128 = head()
        self.out256 = head()
        self.out512 = head()

        self.posenc = SinCosPos2D(64, 64) if include_posenc else None

    def forward(self, z64: torch.Tensor) -> dict[str, torch.Tensor]:
        B = z64.size(0)
        feats = [z64]

        if self.include_posenc:
            # Keep the positional encoding on the same device/dtype as the latent input.
            feats.append(self.posenc(B).to(device=z64.device, dtype=z64.dtype))  # type: ignore[union-attr]

        x = torch.cat(feats, dim=1)
        x = self.in_conv(x)
        x = self.blocks64(x)

        f64 = self.out64(x)

        x = self.up1(x)
        f128 = self.out128(x)

        x = self.up2(x)
        f256 = self.out256(x)

        x = self.up3(x)
        f512 = self.out512(x)

        return {
            "s64": f64,
            "s128": f128,
            "s256": f256,
            "s512": f512,
        }


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Build standard sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of shape [B].
        dim: Output embedding dimension.
        max_period: Controls the lowest frequency.

    Returns:
        Tensor of shape [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    # Pad to the requested dimension when `dim` is odd.
    if dim % 2:
        emb = torch.cat([emb, emb[:, :1]], dim=1)

    return emb


class ResidualBlock2d(nn.Module):
    """
    Basic residual block with optional timestep conditioning.

    Time conditioning is injected after the first conv by projecting the
    timestep embedding to the channel dimension and broadcasting spatially.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(_g(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(_g(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()

        self.use_time = time_dim is not None
        if self.use_time:
            self.time_proj = nn.Linear(time_dim, out_ch)  # type: ignore[assignment]

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))

        if self.use_time:
            # Caller is responsible for providing `t_emb` when time conditioning is enabled.
            h = h + self.time_proj(self.act(t_emb))[:, :, None, None]  # type: ignore[arg-type]

        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class ViTAttention(nn.Module):
    """
    Standard multi-head self-attention operating on [B, N, C] tokens.

    This is implemented directly to avoid an external timm dependency.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, N, C].
        """
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
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
    Apply token attention over a 2D feature map.

    Shape flow:
      [B, C, H, W] -> [B, HW, C] -> attention -> [B, C, H, W]
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_g(8, dim), dim)
        self.attn = ViTAttention(dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        h = h.flatten(2).transpose(1, 2)
        h = self.attn(h)
        h = h.transpose(1, 2).reshape(B, C, H, W)

        return h


class Down(nn.Module):
    """
    Two residual blocks followed by 2x average pooling.

    Optional conditioning is concatenated channel-wise before the first block.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        tdim: int,
        cond_ch: int = 0,
    ):
        super().__init__()
        self.block1 = ResidualBlock2d(in_ch + cond_ch, out_ch, time_dim=tdim)
        self.block2 = ResidualBlock2d(out_ch, out_ch, time_dim=tdim)
        self.down = nn.AvgPool2d(2)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        h = self.block1(x, t_emb)
        h = self.block2(h, t_emb)

        # Return both the skip connection and the downsampled tensor.
        return h, self.down(h)


class Up(nn.Module):
    """
    Bilinear upsample, concatenate skip/conditioning, then apply two residual blocks.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        tdim: int,
        cond_ch: int = 0,
    ):
        super().__init__()
        self.block1 = ResidualBlock2d(in_ch + cond_ch, out_ch, time_dim=tdim)
        self.block2 = ResidualBlock2d(out_ch, out_ch, time_dim=tdim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x


class UNet512(nn.Module):
    """
    Stage-2 512x512 UNet for cascading diffusion.

    Inputs:
      - x: noisy RGB image
      - timesteps: diffusion timestep tensor [B]
      - cond_feats: multi-scale conditioning features from `LatentAdapter`

    Conditioning layout:
      - s512 is injected at the input
      - s256 / s128 are injected into down/up stages
      - s64 is injected at the bottleneck
    """

    def __init__(
        self,
        base_ch: int = 128,
        cond_ch: int = 64,
        time_dim: int = 256,
    ):
        super().__init__()

        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(320, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(3 + cond_ch, ch1, 3, padding=1)

        self.down1 = Down(ch1, ch1, tdim=time_dim, cond_ch=0)
        self.down2 = Down(ch1, ch2, tdim=time_dim, cond_ch=cond_ch)
        self.down3 = Down(ch2, ch3, tdim=time_dim, cond_ch=cond_ch)

        self.attn64 = TimmAttn2D(dim=ch3, num_heads=4)
        self.mid1 = ResidualBlock2d(ch3 + cond_ch, ch3, time_dim=time_dim)
        self.mid2 = ResidualBlock2d(ch3, ch3, time_dim=time_dim)

        self.up3 = Up(ch3 + ch3, ch2, tdim=time_dim, cond_ch=cond_ch)
        self.up2 = Up(ch2 + ch2, ch1, tdim=time_dim, cond_ch=cond_ch)
        self.up1 = Up(ch1 + ch1, ch1, tdim=time_dim, cond_ch=0)

        self.out_norm = nn.GroupNorm(_g(8, ch1), ch1)
        self.out = nn.Conv2d(ch1, 3, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond_feats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy RGB input of shape [B, 3, 512, 512].
            timesteps: Diffusion timestep tensor [B].
            cond_feats: Dict containing s64/s128/s256/s512 feature maps.
        """
        # Build the learned timestep representation once and reuse it throughout the UNet.
        t_emb = self.time_mlp(timestep_embedding(timesteps, 320))

        # Inject the highest-resolution conditioning immediately at the input.
        x = torch.cat([x, cond_feats["s512"]], dim=1)
        x = self.in_conv(x)

        # Down path.
        skip1, x = self.down1(x, t_emb, cond=None)
        skip2, x = self.down2(x, t_emb, cond_feats["s256"])
        skip3, x = self.down3(x, t_emb, cond_feats["s128"])

        # Bottleneck.
        x = self.attn64(x)
        x = torch.cat([x, cond_feats["s64"]], dim=1)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # Up path.
        x = self.up3(x, skip3, t_emb, cond_feats["s128"])
        x = self.up2(x, skip2, t_emb, cond_feats["s256"])
        x = self.up1(x, skip1, t_emb, cond=None)

        # Clamp normalized activations before the final projection to reduce extreme values.
        x = self.out(self.out_norm(x).clamp(-6, 6))

        # Bound output to image range expected by the diffusion objective.
        return torch.tanh(x)