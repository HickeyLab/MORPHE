from __future__ import annotations

import torch

from core.latent_diffusion.model import (
    BBoxEncoder,
    CondEncoder,
    CondEncoder3D,
    CoordEncoder,
    PositionalEncoding2D,
    ResidualBlock,
    Transpose,
)


# ── Transpose ───────────────────────────────────────────────────────


def test_transpose_swaps_dims() -> None:
    layer = Transpose(1, 2)
    x = torch.randn(2, 3, 4)

    result = layer(x)

    assert result.shape == (2, 4, 3)


# ── PositionalEncoding2D ────────────────────────────────────────────


def test_positional_encoding_adds_to_input() -> None:
    pe = PositionalEncoding2D(num_patches=16, dim=32)
    x = torch.zeros(1, 16, 32)

    result = pe(x)

    assert result.shape == (1, 16, 32)
    # should not be all zeros since encoding is added
    assert not torch.all(result == 0)


def test_positional_encoding_handles_shorter_sequence() -> None:
    pe = PositionalEncoding2D(num_patches=16, dim=32)
    x = torch.zeros(2, 8, 32)

    result = pe(x)

    assert result.shape == (2, 8, 32)


# ── ResidualBlock ───────────────────────────────────────────────────


def test_residual_block_same_channels() -> None:
    block = ResidualBlock(in_channels=64, out_channels=64)
    x = torch.randn(2, 64, 8, 8)

    result = block(x)

    assert result.shape == (2, 64, 8, 8)


def test_residual_block_different_channels() -> None:
    block = ResidualBlock(in_channels=32, out_channels=64)
    x = torch.randn(2, 32, 8, 8)

    result = block(x)

    assert result.shape == (2, 64, 8, 8)


# ── CondEncoder ─────────────────────────────────────────────────────


def test_cond_encoder_output_shape() -> None:
    enc = CondEncoder(in_channels=4, out_channels=64, num_tokens=16)
    x = torch.randn(2, 4, 64, 64)

    result = enc(x)

    assert result.shape == (2, 16, 64)


# ── CondEncoder3D ───────────────────────────────────────────────────


def test_cond_encoder_3d_output_shape() -> None:
    enc = CondEncoder3D(in_channels=4, out_channels=64, num_tokens=16)
    x = torch.randn(2, 4, 64, 64)

    result = enc(x)

    assert result.shape == (2, 16, 64)


# ── CoordEncoder ────────────────────────────────────────────────────


def test_coord_encoder_output_shape() -> None:
    enc = CoordEncoder(embed_dim=32, num_tokens=16)
    mask = torch.randn(2, 1, 128, 128)

    result = enc(mask)

    assert result.shape == (2, 16, 32)


# ── BBoxEncoder ─────────────────────────────────────────────────────


def test_bbox_encoder_output_shape() -> None:
    enc = BBoxEncoder(in_dim=4, out_dim=32)
    bbox = torch.randn(2, 4)

    result = enc(bbox)

    assert result.shape == (2, 32)


def test_bbox_encoder_custom_dims() -> None:
    enc = BBoxEncoder(in_dim=6, out_dim=64)
    bbox = torch.randn(3, 6)

    result = enc(bbox)

    assert result.shape == (3, 64)
