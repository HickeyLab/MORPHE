from __future__ import annotations

import torch

from core.pixel_diffusion.models import (
    LatentAdapter,
    UNet512,
    SinCosPos2D,
    ResInceptionDilated,
    UpStage,
    ResidualBlock2d,
    ViTAttention,
    TimmAttn2D,
    Down,
    Up,
    timestep_embedding,
)


# ── SinCosPos2D ─────────────────────────────────────────────────────


def test_sincos_pos2d_shape() -> None:
    pe = SinCosPos2D(H=32, W=32)
    result = pe(B=2)
    assert result.shape == (2, 4, 32, 32)


def test_sincos_pos2d_default_shape() -> None:
    pe = SinCosPos2D()
    result = pe(B=1)
    assert result.shape == (1, 4, 64, 64)


# ── ResInceptionDilated ─────────────────────────────────────────────


def test_res_inception_preserves_shape() -> None:
    block = ResInceptionDilated(ch=64)
    x = torch.randn(2, 64, 16, 16)
    result = block(x)
    assert result.shape == (2, 64, 16, 16)


# ── UpStage ─────────────────────────────────────────────────────────


def test_upstage_doubles_resolution() -> None:
    up = UpStage(ch=64)
    x = torch.randn(2, 64, 8, 8)
    result = up(x)
    assert result.shape == (2, 64, 16, 16)


# ── LatentAdapter ───────────────────────────────────────────────────


def test_latent_adapter_output_keys_and_shapes() -> None:
    adapter = LatentAdapter(cz=4, cond_ch=32, width=64, num_blocks_64=1)
    z = torch.randn(2, 4, 64, 64)

    feats = adapter(z)

    assert set(feats.keys()) == {"s64", "s128", "s256", "s512"}
    assert feats["s64"].shape == (2, 32, 64, 64)
    assert feats["s128"].shape == (2, 32, 128, 128)
    assert feats["s256"].shape == (2, 32, 256, 256)
    assert feats["s512"].shape == (2, 32, 512, 512)


def test_latent_adapter_without_posenc() -> None:
    adapter = LatentAdapter(cz=4, cond_ch=32, width=64, num_blocks_64=1, include_posenc=False)
    z = torch.randn(2, 4, 64, 64)

    feats = adapter(z)

    assert set(feats.keys()) == {"s64", "s128", "s256", "s512"}
    assert feats["s64"].shape[0] == 2


# ── timestep_embedding ──────────────────────────────────────────────


def test_timestep_embedding_shape_even() -> None:
    t = torch.tensor([0, 50, 100])
    emb = timestep_embedding(t, dim=64)
    assert emb.shape == (3, 64)


def test_timestep_embedding_shape_odd() -> None:
    t = torch.tensor([0, 50])
    emb = timestep_embedding(t, dim=65)
    assert emb.shape == (2, 65)


# ── ResidualBlock2d ─────────────────────────────────────────────────


def test_residual_block_2d_same_channels() -> None:
    block = ResidualBlock2d(in_ch=64, out_ch=64)
    x = torch.randn(2, 64, 16, 16)
    result = block(x)
    assert result.shape == (2, 64, 16, 16)


def test_residual_block_2d_different_channels() -> None:
    block = ResidualBlock2d(in_ch=32, out_ch=64)
    x = torch.randn(2, 32, 16, 16)
    result = block(x)
    assert result.shape == (2, 64, 16, 16)


def test_residual_block_2d_with_time() -> None:
    block = ResidualBlock2d(in_ch=64, out_ch=64, time_dim=128)
    x = torch.randn(2, 64, 16, 16)
    t_emb = torch.randn(2, 128)
    result = block(x, t_emb)
    assert result.shape == (2, 64, 16, 16)


# ── ViTAttention ────────────────────────────────────────────────────


def test_vit_attention_shape() -> None:
    attn = ViTAttention(dim=64, num_heads=4)
    x = torch.randn(2, 16, 64)
    result = attn(x)
    assert result.shape == (2, 16, 64)


# ── TimmAttn2D ──────────────────────────────────────────────────────


def test_timm_attn_2d_preserves_shape() -> None:
    attn = TimmAttn2D(dim=64, num_heads=4)
    x = torch.randn(2, 64, 8, 8)
    result = attn(x)
    assert result.shape == (2, 64, 8, 8)


# ── Down ────────────────────────────────────────────────────────────


def test_down_halves_resolution() -> None:
    down = Down(in_ch=64, out_ch=128, tdim=256)
    x = torch.randn(2, 64, 32, 32)
    t_emb = torch.randn(2, 256)

    skip, downsampled = down(x, t_emb)

    assert skip.shape == (2, 128, 32, 32)
    assert downsampled.shape == (2, 128, 16, 16)


def test_down_with_conditioning() -> None:
    down = Down(in_ch=64, out_ch=128, tdim=256, cond_ch=32)
    x = torch.randn(2, 64, 32, 32)
    t_emb = torch.randn(2, 256)
    cond = torch.randn(2, 32, 32, 32)

    skip, downsampled = down(x, t_emb, cond)

    assert skip.shape == (2, 128, 32, 32)
    assert downsampled.shape == (2, 128, 16, 16)


# ── Up ──────────────────────────────────────────────────────────────


def test_up_doubles_resolution() -> None:
    up = Up(in_ch=256, out_ch=128, tdim=256)
    x = torch.randn(2, 128, 16, 16)
    skip = torch.randn(2, 128, 32, 32)
    t_emb = torch.randn(2, 256)

    result = up(x, skip, t_emb)

    assert result.shape == (2, 128, 32, 32)


def test_up_with_conditioning() -> None:
    up = Up(in_ch=256, out_ch=128, tdim=256, cond_ch=32)
    x = torch.randn(2, 128, 16, 16)
    skip = torch.randn(2, 128, 32, 32)
    t_emb = torch.randn(2, 256)
    cond = torch.randn(2, 32, 32, 32)

    result = up(x, skip, t_emb, cond)

    assert result.shape == (2, 128, 32, 32)


# ── UNet512 full forward ────────────────────────────────────────────


def test_unet512_forward_shape() -> None:
    unet = UNet512(base_ch=32, cond_ch=16, time_dim=64)
    x = torch.randn(1, 3, 512, 512)
    timesteps = torch.tensor([50])
    cond_feats = {
        "s64": torch.randn(1, 16, 64, 64),
        "s128": torch.randn(1, 16, 128, 128),
        "s256": torch.randn(1, 16, 256, 256),
        "s512": torch.randn(1, 16, 512, 512),
    }

    result = unet(x, timesteps, cond_feats)

    assert result.shape == (1, 3, 512, 512)
    # output should be bounded by tanh
    assert result.min() >= -1.0
    assert result.max() <= 1.0
