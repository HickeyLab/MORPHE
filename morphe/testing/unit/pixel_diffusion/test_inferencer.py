from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.pixel_diffusion.inferencer import PixelDiffusionInferencer


# ── _validate_decode_inputs ─────────────────────────────────────────


def test_validate_decode_inputs_raises_when_both_none() -> None:
    with pytest.raises(ValueError, match="Provide either"):
        PixelDiffusionInferencer._validate_decode_inputs(latent=None, latent_path=None)


def test_validate_decode_inputs_raises_when_both_provided() -> None:
    with pytest.raises(ValueError, match="Provide only one"):
        PixelDiffusionInferencer._validate_decode_inputs(
            latent=torch.randn(4, 64, 64),
            latent_path="/some/path.pt",
        )


def test_validate_decode_inputs_raises_for_missing_path() -> None:
    with pytest.raises(FileNotFoundError, match="latent_path does not exist"):
        PixelDiffusionInferencer._validate_decode_inputs(
            latent=None,
            latent_path="/nonexistent/path.pt",
        )


def test_validate_decode_inputs_accepts_latent_only() -> None:
    PixelDiffusionInferencer._validate_decode_inputs(
        latent=torch.randn(4, 64, 64),
        latent_path=None,
    )  # should not raise


def test_validate_decode_inputs_accepts_path_only(tmp_path: Path) -> None:
    path = tmp_path / "latent.pt"
    torch.save(torch.randn(4, 64, 64), path)

    PixelDiffusionInferencer._validate_decode_inputs(
        latent=None,
        latent_path=path,
    )  # should not raise


# ── _load_latent ────────────────────────────────────────────────────


def test_load_latent_returns_tensor_directly() -> None:
    t = torch.randn(4, 64, 64)
    result = PixelDiffusionInferencer._load_latent(latent=t, latent_path=None)
    assert result is t


def test_load_latent_raises_for_non_tensor() -> None:
    with pytest.raises(TypeError, match="latent must be a torch.Tensor"):
        PixelDiffusionInferencer._load_latent(latent="bad", latent_path=None)


def test_load_latent_from_file(tmp_path: Path) -> None:
    expected = torch.randn(4, 64, 64)
    path = tmp_path / "latent.pt"
    torch.save(expected, path)

    result = PixelDiffusionInferencer._load_latent(latent=None, latent_path=path)
    assert torch.equal(result, expected)


def test_load_latent_raises_when_file_not_tensor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(b"x")

    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"not": "tensor"})

    with pytest.raises(TypeError, match="Loaded latent is not a torch.Tensor"):
        PixelDiffusionInferencer._load_latent(latent=None, latent_path=path)


def test_load_latent_raises_when_both_none() -> None:
    with pytest.raises(ValueError, match="latent_path must be provided"):
        PixelDiffusionInferencer._load_latent(latent=None, latent_path=None)


# ── _ensure_batched_latent ──────────────────────────────────────────


def test_ensure_batched_latent_adds_batch_dim_for_3d() -> None:
    t = torch.randn(4, 64, 64)
    result = PixelDiffusionInferencer._ensure_batched_latent(t)
    assert result.shape == (1, 4, 64, 64)


def test_ensure_batched_latent_passes_through_4d() -> None:
    t = torch.randn(2, 4, 64, 64)
    result = PixelDiffusionInferencer._ensure_batched_latent(t)
    assert result.shape == (2, 4, 64, 64)


def test_ensure_batched_latent_rejects_2d() -> None:
    with pytest.raises(ValueError, match="ndim"):
        PixelDiffusionInferencer._ensure_batched_latent(torch.randn(64, 64))


def test_ensure_batched_latent_rejects_5d() -> None:
    with pytest.raises(ValueError, match="ndim"):
        PixelDiffusionInferencer._ensure_batched_latent(torch.randn(1, 1, 4, 64, 64))
