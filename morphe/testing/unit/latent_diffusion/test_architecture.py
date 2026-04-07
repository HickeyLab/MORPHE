from __future__ import annotations

import pytest

from core.latent_diffusion.architecture import LatentArchitectureSpec


# ── to_dict / from_dict round-trip ──────────────────────────────────


def test_to_dict_from_dict_round_trip_defaults() -> None:
    spec = LatentArchitectureSpec()

    payload = spec.to_dict()
    reconstructed = LatentArchitectureSpec.from_dict(payload)

    assert reconstructed.unet_pretrained_path == spec.unet_pretrained_path
    assert reconstructed.ae_pretrained_path == spec.ae_pretrained_path
    assert reconstructed.scheduler_pretrained_path == spec.scheduler_pretrained_path
    assert reconstructed.cond_encoder_type == spec.cond_encoder_type
    assert reconstructed.coord_encoder_type == spec.coord_encoder_type
    assert reconstructed.bbox_encoder_type == spec.bbox_encoder_type


def test_to_dict_from_dict_round_trip_with_optional_encoders() -> None:
    spec = LatentArchitectureSpec(
        cond_encoder_type="cond3d",
        coord_encoder_type="coord",
        bbox_encoder_type="bbox",
        cond_encoder_kwargs={"in_channels": 4, "out_channels": 768},
        coord_encoder_kwargs={"embed_dim": 32},
        bbox_encoder_kwargs={"in_dim": 4, "out_dim": 32},
    )

    payload = spec.to_dict()
    reconstructed = LatentArchitectureSpec.from_dict(payload)

    assert reconstructed.cond_encoder_type == "cond3d"
    assert reconstructed.coord_encoder_type == "coord"
    assert reconstructed.bbox_encoder_type == "bbox"
    assert dict(reconstructed.cond_encoder_kwargs) == {"in_channels": 4, "out_channels": 768}
    assert dict(reconstructed.coord_encoder_kwargs) == {"embed_dim": 32}
    assert dict(reconstructed.bbox_encoder_kwargs) == {"in_dim": 4, "out_dim": 32}


def test_to_dict_returns_plain_dict() -> None:
    spec = LatentArchitectureSpec()
    payload = spec.to_dict()

    assert isinstance(payload, dict)
    expected_keys = {
        "unet_pretrained_path",
        "ae_pretrained_path",
        "scheduler_pretrained_path",
        "cond_encoder_type",
        "coord_encoder_type",
        "bbox_encoder_type",
        "cond_encoder_kwargs",
        "coord_encoder_kwargs",
        "bbox_encoder_kwargs",
    }
    assert set(payload.keys()) == expected_keys


def test_to_dict_none_encoder_types_preserved() -> None:
    spec = LatentArchitectureSpec(
        coord_encoder_type=None,
        bbox_encoder_type=None,
    )
    payload = spec.to_dict()
    reconstructed = LatentArchitectureSpec.from_dict(payload)

    assert reconstructed.coord_encoder_type is None
    assert reconstructed.bbox_encoder_type is None


# ── from_dict validation ────────────────────────────────────────────


def test_from_dict_raises_on_missing_key() -> None:
    payload = {
        "unet_pretrained_path": "path",
        "ae_pretrained_path": "path",
        # scheduler_pretrained_path intentionally missing
        "cond_encoder_type": "cond",
        "coord_encoder_type": None,
        "bbox_encoder_type": None,
        "cond_encoder_kwargs": {},
        "coord_encoder_kwargs": {},
        "bbox_encoder_kwargs": {},
    }

    with pytest.raises(ValueError, match="missing key 'scheduler_pretrained_path'"):
        LatentArchitectureSpec.from_dict(payload)


# ── _build_*_encoder validation ─────────────────────────────────────


def test_build_cond_encoder_raises_for_unknown_type() -> None:
    import torch

    spec = LatentArchitectureSpec(cond_encoder_type="unknown")

    with pytest.raises(ValueError, match="Unknown cond_encoder_type"):
        spec._build_cond_encoder(device=torch.device("cpu"), dtype=torch.float32)


def test_build_coord_encoder_returns_none_when_type_is_none() -> None:
    import torch

    spec = LatentArchitectureSpec(coord_encoder_type=None)

    result = spec._build_coord_encoder(device=torch.device("cpu"), dtype=torch.float32)
    assert result is None


def test_build_coord_encoder_raises_for_unknown_type() -> None:
    import torch

    spec = LatentArchitectureSpec(coord_encoder_type="unknown")

    with pytest.raises(ValueError, match="Unknown coord_encoder_type"):
        spec._build_coord_encoder(device=torch.device("cpu"), dtype=torch.float32)


def test_build_bbox_encoder_returns_none_when_type_is_none() -> None:
    import torch

    spec = LatentArchitectureSpec(bbox_encoder_type=None)

    result = spec._build_bbox_encoder(device=torch.device("cpu"), dtype=torch.float32)
    assert result is None


def test_build_bbox_encoder_raises_for_unknown_type() -> None:
    import torch

    spec = LatentArchitectureSpec(bbox_encoder_type="unknown")

    with pytest.raises(ValueError, match="Unknown bbox_encoder_type"):
        spec._build_bbox_encoder(device=torch.device("cpu"), dtype=torch.float32)
