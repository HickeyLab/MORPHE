from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from core.latent_diffusion.artifact import LatentDiffusionArtifact
from core.latent_diffusion.architecture import LatentArchitectureSpec
from constants import InferenceMode


def _make_spec() -> LatentArchitectureSpec:
    return LatentArchitectureSpec(
        cond_encoder_type="cond",
        coord_encoder_type=None,
        bbox_encoder_type=None,
    )


def _make_artifact(
    *,
    inference_mode: InferenceMode = InferenceMode.INPAINTING,
    coord_encoder_state_dict: dict[str, torch.Tensor] | None = None,
    bbox_encoder_state_dict: dict[str, torch.Tensor] | None = None,
    img_size: int | None = 256,
) -> LatentDiffusionArtifact:
    return LatentDiffusionArtifact(
        unet_state_dict={"weight": torch.randn(4, 4)},
        cond_encoder_state_dict={"weight": torch.randn(3, 3)},
        coord_encoder_state_dict=coord_encoder_state_dict,
        bbox_encoder_state_dict=bbox_encoder_state_dict,
        architecture=_make_spec(),
        inference_mode=inference_mode,
        img_size=img_size,
    )


def _make_payload(artifact: LatentDiffusionArtifact) -> dict[str, object]:
    return {
        "unet_state_dict": artifact.unet_state_dict,
        "cond_encoder_state_dict": artifact.cond_encoder_state_dict,
        "coord_encoder_state_dict": artifact.coord_encoder_state_dict,
        "bbox_encoder_state_dict": artifact.bbox_encoder_state_dict,
        "architecture": artifact.architecture.to_dict(),
        "inference_mode": artifact.inference_mode.value,
        "img_size": artifact.img_size,
    }


# ── save / load round-trip ──────────────────────────────────────────


def test_save_and_load_round_trip_preserves_metadata(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "nested" / "artifact.pt"

    artifact.save(save_path)
    assert save_path.exists()

    loaded = LatentDiffusionArtifact.load(save_path)

    assert loaded.inference_mode == artifact.inference_mode
    assert loaded.img_size == artifact.img_size
    assert loaded.coord_encoder_state_dict is None
    assert loaded.bbox_encoder_state_dict is None

    for key, expected in artifact.unet_state_dict.items():
        assert torch.equal(loaded.unet_state_dict[key], expected.cpu())

    for key, expected in artifact.cond_encoder_state_dict.items():
        assert torch.equal(loaded.cond_encoder_state_dict[key], expected.cpu())


def test_save_and_load_round_trip_with_optional_encoders(tmp_path: Path) -> None:
    artifact = _make_artifact(
        coord_encoder_state_dict={"w": torch.randn(2, 2)},
        bbox_encoder_state_dict={"w": torch.randn(3, 3)},
    )
    save_path = tmp_path / "artifact.pt"

    artifact.save(save_path)
    loaded = LatentDiffusionArtifact.load(save_path)

    assert loaded.coord_encoder_state_dict is not None
    assert torch.equal(
        loaded.coord_encoder_state_dict["w"],
        artifact.coord_encoder_state_dict["w"].cpu(),
    )
    assert loaded.bbox_encoder_state_dict is not None
    assert torch.equal(
        loaded.bbox_encoder_state_dict["w"],
        artifact.bbox_encoder_state_dict["w"].cpu(),
    )


def test_save_serializes_cpu_tensors(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "artifact.pt"
    artifact.save(save_path)

    payload = torch.load(save_path, map_location="cpu", weights_only=True)

    for tensor in payload["unet_state_dict"].values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"

    for tensor in payload["cond_encoder_state_dict"].values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"


# ── load validation ─────────────────────────────────────────────────


def test_load_raises_for_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(b"x")

    monkeypatch.setattr(torch, "load", lambda *a, **kw: ["not", "a", "dict"])

    with pytest.raises(ValueError, match="payload must be a dictionary"):
        LatentDiffusionArtifact.load(path)


def test_load_raises_when_required_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.pt"
    path.write_bytes(b"x")

    payload = {
        "unet_state_dict": {},
        "cond_encoder_state_dict": {},
        "coord_encoder_state_dict": None,
        "bbox_encoder_state_dict": None,
        "architecture": {},
        "inference_mode": "inpainting",
        # img_size intentionally missing
    }

    monkeypatch.setattr(torch, "load", lambda *a, **kw: payload)

    with pytest.raises(ValueError, match="missing key 'img_size'"):
        LatentDiffusionArtifact.load(path)


def test_load_raises_when_architecture_not_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad_arch.pt"
    path.write_bytes(b"x")

    payload = {
        "unet_state_dict": {},
        "cond_encoder_state_dict": {},
        "coord_encoder_state_dict": None,
        "bbox_encoder_state_dict": None,
        "architecture": "not_a_dict",
        "inference_mode": "inpainting",
        "img_size": 256,
    }

    monkeypatch.setattr(torch, "load", lambda *a, **kw: payload)

    with pytest.raises(ValueError, match="'architecture' must be a dictionary"):
        LatentDiffusionArtifact.load(path)


def test_load_falls_back_when_weights_only_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "fallback.pt"
    path.write_bytes(b"x")

    artifact = _make_artifact()
    payload = _make_payload(artifact)
    calls: list[dict[str, object]] = []

    def fake_load(*args, **kwargs):
        calls.append(dict(kwargs))
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword")
        return payload

    monkeypatch.setattr(torch, "load", fake_load)

    loaded = LatentDiffusionArtifact.load(path)

    assert isinstance(loaded, LatentDiffusionArtifact)
    assert len(calls) == 2
    assert "weights_only" in calls[0]
    assert "weights_only" not in calls[1]


# ── _state_dict_to_cpu ──────────────────────────────────────────────


def test_state_dict_to_cpu_returns_none_for_none() -> None:
    assert LatentDiffusionArtifact._state_dict_to_cpu(None) is None


def test_state_dict_to_cpu_detaches_and_moves_to_cpu() -> None:
    sd = {"w": torch.randn(2, 2, requires_grad=True)}
    result = LatentDiffusionArtifact._state_dict_to_cpu(sd)

    assert result is not None
    assert result["w"].device.type == "cpu"
    assert not result["w"].requires_grad
