import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL # type: ignore

class Transpose(nn.Module):
    """
    Module wrapper for `torch.Tensor.transpose`.

    This is useful for inserting transpose operations inside
    `nn.Sequential` pipelines.

    Args:
        dim0: First dimension to swap.
        dim1: Second dimension to swap.
    """

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
        """
        Apply transpose along the specified dimensions.

        Args:
            x: Input tensor.

        Returns:
            Transposed tensor.
        """
        return x.transpose(self.dim0, self.dim1)


class PositionalEncoding2D(nn.Module):
    """
    Fixed sinusoidal positional encoding for token sequences.

    This encoding is added to token embeddings to inject spatial
    information. Despite the name, this implementation operates
    on flattened 2D patches (sequence form).

    Args:
        num_patches: Number of tokens (sequence length).
        dim: Embedding dimension per token.
    """

    def __init__(
        self, 
        num_patches: int, 
        dim: int
    ):
        super().__init__()
        self.register_buffer(
            'pos_embed',
            self.build_sincos_encoding(num_patches, dim),
            persistent=False
        )

    def build_sincos_encoding(
        self, 
        num_patches: int, 
        dim: int
    ):
        """
        Build sinusoidal positional encoding matrix.

        Args:
            num_patches: Number of positions.
            dim: Embedding dimension.

        Returns:
            Tensor of shape (1, num_patches, dim).
        """
        pe = torch.zeros(num_patches, dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """
        Add positional encoding to token embeddings.

        Args:
            x: Input tensor of shape (B, num_tokens, dim).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pos_embed[:, :x.size(1), :]


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with GroupNorm and SiLU activation.

    Structure:
        Conv → GN → SiLU → Conv → GN + skip connection

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

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
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        """
        Apply residual block.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Output tensor (B, out_channels, H, W).
        """
        return self.block(x) + self.skip(x)


class CondEncoder(nn.Module):
    """
    Conditioning encoder that converts latent feature maps into token sequences.

    Pipeline:
        CNN encoder → flatten spatial → tokens → positional encoding → normalization

    Used for:
        - Conditioning UNet in latent diffusion models

    Args:
        in_channels: Input latent channels.
        out_channels: Token embedding dimension.
        num_tokens: Number of output tokens.
    """

    def __init__(
        self, 
        in_channels: int = 4, 
        out_channels: int = 736, 
        num_tokens: int = 64
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64),
            nn.AvgPool2d(2),
            ResidualBlock(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock(128, 256),
            nn.AvgPool2d(2),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(2),
            Transpose(-1, -2),
        )
        self.pos_embed = PositionalEncoding2D(num_patches=num_tokens, dim=out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        """
        Encode latent tensor into conditioning tokens.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Token tensor (B, num_tokens, out_channels).
        """
        feat = self.encoder(x)
        tokens = self.proj(feat)
        tokens = self.pos_embed(tokens)
        tokens = self.norm(tokens)
        return tokens


class CoordEncoder(nn.Module):
    """
    Encoder for spatial masks into token embeddings.

    Converts a binary mask into a sequence of tokens using
    an MLP over flattened spatial input.

    Args:
        embed_dim: Dimension of each output token.
        num_tokens: Number of tokens to generate.
    """

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
            nn.Linear(2048, num_tokens * embed_dim)
        )

    def forward(
        self, 
        mask: torch.Tensor
    ):
        """
        Encode mask into token sequence.

        Args:
            mask: Input mask tensor (B, 1, H, W).

        Returns:
            Token tensor (B, num_tokens, embed_dim).
        """
        mask_ds = F.interpolate(mask, size=(64, 64), mode="nearest")
        B = mask_ds.shape[0]
        x = mask_ds.view(B, -1)
        x = self.mlp(x)
        x = x.view(B, self.num_tokens, self.embed_dim)
        return x


class CondEncoder3D(nn.Module):
    """
    Conditioning encoder for 3D-aware tasks.

    Architecturally identical to `CondEncoder`, but conceptually used
    for encoding context across multiple frames or slices.

    Args:
        in_channels: Input latent channels.
        out_channels: Token embedding dimension.
        num_tokens: Number of output tokens.
    """

    def __init__(
        self,
        in_channels: int = 4, 
        out_channels: int = 768, 
        num_tokens: int = 64
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64),
            nn.AvgPool2d(2),
            ResidualBlock(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock(128, 256),
            nn.AvgPool2d(2),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(2),
            Transpose(-1, -2),
        )
        self.pos_embed = PositionalEncoding2D(num_patches=num_tokens, dim=out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        """
        Encode latent tensor into conditioning tokens.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Token tensor (B, num_tokens, out_channels).
        """
        feat = self.encoder(x)
        tokens = self.proj(feat)
        tokens = self.pos_embed(tokens)
        tokens = self.norm(tokens)
        return tokens


class BBoxEncoder(nn.Module):
    """
    Encoder for bounding box coordinates.

    Maps normalized bounding box coordinates into a learned embedding
    space for conditioning diffusion models.

    Args:
        in_dim: Input dimension (typically 4 for [x1, y1, x2, y2]).
        out_dim: Output embedding dimension.
    """

    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 32,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Encode bounding boxes.

        Args:
            bbox: Tensor of shape (B, 4).

        Returns:
            Encoded tensor of shape (B, out_dim).
        """
        return self.encoder(bbox)
    

class VAEEncoder(nn.Module):
    """
    Wrapper around Stable Diffusion VAE for latent encoding.

    This module:
    - Loads a pretrained VAE
    - Freezes all parameters
    - Provides a simple `encode` interface
    - Applies the correct latent scaling factor

    Args:
        pretrained_path: HuggingFace model path for the VAE.
        device: Device to load the model on.
    """

    def __init__(
        self, 
        pretrained_path: str = "runwayml/stable-diffusion-v1-5", 
        device: torch.device | str | None = None
    ):
        super().__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            pretrained_path, subfolder="vae"
        ).to(self.device) # type: ignore

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        config = self.vae.config
        if isinstance(config, dict):
            self.scaling = config.get("scaling_factor", 0.18215)
        else:
            self.scaling = getattr(config, "scaling_factor", 0.18215)

    @torch.no_grad()
    def encode(self, imgs):
        """
        Encode images into latent space.

        Args:
            imgs: Input tensor (B, 3, H, W), typically normalized to [-1, 1].

        Returns:
            Latent tensor (B, 4, H/8, W/8).
        """
        latents = self.vae.encode(imgs).latent_dist.sample() # type: ignore
        latents = latents * self.scaling
        return latents