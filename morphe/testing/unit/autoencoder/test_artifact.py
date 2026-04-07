from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from core.autoencoder.artifact import AutoencoderArtifact
from core.autoencoder.model import Autoencoder


def _make_artifact(
    *,
    in_dim: int = 5,
    bottleneck_dim: int = 3,
    hidden_dim: int = 16,
) -> AutoencoderArtifact:
    model = Autoencoder(
        in_dim=in_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
    ).to(device="cpu", dtype=torch.float32)

    return AutoencoderArtifact(
        state_dict=model.state_dict(),
        input_cols=("f1", "f2", "f3", "f4", "f5"),
        in_dim=in_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        z_min=torch.tensor([0.0, 0.0, 0.0]),
        z_max=torch.tensor([1.0, 1.0, 1.0]),
    )


def _make_payload(artifact: AutoencoderArtifact) -> dict[str, object]:
    return {
        "state_dict": artifact.state_dict,
        "input_cols": artifact.input_cols,
        "in_dim": artifact.in_dim,
        "bottleneck_dim": artifact.bottleneck_dim,
        "hidden_dim": artifact.hidden_dim,
        "z_min": artifact.z_min,
        "z_max": artifact.z_max,
    }


def test_build_model_loads_weights_and_sets_eval_cpu_float32() -> None:
    artifact = _make_artifact()

    model = artifact.build_model(device="cpu", dtype=torch.float32)

    assert model.training is False
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float32

    for key, expected in artifact.state_dict.items():
        actual = model.state_dict()[key]
        assert torch.equal(actual, expected)


def test_build_model_reconstructs_matching_state_dict_shapes() -> None:
    artifact = _make_artifact(in_dim=8, bottleneck_dim=3, hidden_dim=32)

    model = artifact.build_model(device="cpu", dtype=torch.float32)
    rebuilt_state_dict = model.state_dict()

    assert rebuilt_state_dict.keys() == artifact.state_dict.keys()

    for key, expected in artifact.state_dict.items():
        assert rebuilt_state_dict[key].shape == expected.shape


def test_save_and_load_round_trip_preserves_metadata_and_weights(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "nested" / "autoencoder_artifact.pt"

    artifact.save(save_path)

    assert save_path.exists()

    loaded = AutoencoderArtifact.load(save_path)

    assert loaded.in_dim == artifact.in_dim
    assert loaded.bottleneck_dim == artifact.bottleneck_dim
    assert loaded.hidden_dim == artifact.hidden_dim
    assert loaded.input_cols == artifact.input_cols
    assert torch.equal(loaded.z_min, artifact.z_min)
    assert torch.equal(loaded.z_max, artifact.z_max)

    assert isinstance(loaded.state_dict, Mapping)

    for key, expected in artifact.state_dict.items():
        actual = loaded.state_dict[key]
        assert actual.device.type == "cpu"
        assert torch.equal(actual, expected.cpu())


def test_save_serializes_cpu_state_dict_tensors(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "artifact.pt"

    artifact.save(save_path)

    payload = torch.load(save_path, map_location="cpu", weights_only=True)

    assert isinstance(payload, Mapping)
    assert "state_dict" in payload
    assert isinstance(payload["state_dict"], Mapping)

    for tensor in payload["state_dict"].values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"


def test_load_raises_value_error_when_required_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_key.pt"
    path.write_bytes(b"x")

    payload = {
        "state_dict": {},
        "input_cols": ("f1",),
        "in_dim": 5,
        "bottleneck_dim": 3,
        "hidden_dim": 16,
        "z_min": torch.tensor([0.0]),
        # z_max intentionally missing
    }

    def fake_load(*args, **kwargs):
        return payload

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(ValueError, match="missing keys"):
        AutoencoderArtifact.load(path)


def test_load_falls_back_when_torch_load_rejects_weights_only(
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

    loaded = AutoencoderArtifact.load(path)

    assert isinstance(loaded, AutoencoderArtifact)
    assert len(calls) == 2
    assert "weights_only" in calls[0]
    assert "weights_only" not in calls[1]
