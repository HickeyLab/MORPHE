from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from core.gcnn.artifact import GCNNArtifact
from core.gcnn.inferencer import GCNNInferencer
from core.gcnn.model import GCNClassifier


def _make_artifact(*, hidden_channels: int = 5, num_classes: int = 2) -> GCNNArtifact:
    base_model = GCNClassifier(
        in_channels=3,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=0.1,
        alpha=0.9,
        K=2,
    ).to(device="cpu", dtype=torch.float32)

    return GCNNArtifact(
        model_state_dict=base_model.state_dict(),
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=0.1,
        alpha=0.9,
        K=2,
        k_neighbors=4,
        label_col="label",
        feature_cols=("f1", "f2", "f3"),
        region_col="region",
        pos_cols=("x", "y"),
        classes_=("a", "b"),
    )


def _make_payload(artifact: GCNNArtifact) -> dict[str, object]:
    return {
        "model_state_dict": artifact.model_state_dict,
        "hidden_channels": artifact.hidden_channels,
        "num_classes": artifact.num_classes,
        "dropout": artifact.dropout,
        "alpha": artifact.alpha,
        "K": artifact.K,
        "k_neighbors": artifact.k_neighbors,
        "label_col": artifact.label_col,
        "feature_cols": artifact.feature_cols,
        "region_col": artifact.region_col,
        "pos_cols": artifact.pos_cols,
        "classes_": artifact.classes_,
    }


def test_build_model_loads_weights_and_sets_eval_cpu_float32() -> None:
    artifact = _make_artifact()

    model = artifact.build_model(device="cpu", dtype=torch.float32)

    assert model.training is False
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float32

    for key, expected in artifact.model_state_dict.items():
        actual = model.state_dict()[key]
        assert torch.equal(actual, expected)


def test_build_model_reconstructs_matching_state_dict_shapes() -> None:
    artifact = _make_artifact(hidden_channels=7, num_classes=3)

    model = artifact.build_model(device="cpu", dtype=torch.float32)
    rebuilt_state_dict = model.state_dict()

    assert rebuilt_state_dict.keys() == artifact.model_state_dict.keys()

    for key, expected in artifact.model_state_dict.items():
        assert rebuilt_state_dict[key].shape == expected.shape


def test_save_and_load_round_trip_preserves_metadata_and_weights(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "nested" / "gcnn_artifact.pt"

    artifact.save(save_path)

    assert save_path.exists()

    loaded = GCNNArtifact.load(save_path)

    assert loaded.hidden_channels == artifact.hidden_channels
    assert loaded.num_classes == artifact.num_classes
    assert loaded.dropout == artifact.dropout
    assert loaded.alpha == artifact.alpha
    assert loaded.K == artifact.K
    assert loaded.k_neighbors == artifact.k_neighbors
    assert loaded.label_col == artifact.label_col
    assert loaded.feature_cols == artifact.feature_cols
    assert loaded.region_col == artifact.region_col
    assert loaded.pos_cols == artifact.pos_cols
    assert loaded.classes_ == artifact.classes_

    assert isinstance(loaded.model_state_dict, Mapping)
    assert isinstance(loaded.feature_cols, tuple)
    assert isinstance(loaded.pos_cols, tuple)
    assert isinstance(loaded.classes_, tuple)

    for key, expected in artifact.model_state_dict.items():
        actual = loaded.model_state_dict[key]
        assert actual.device.type == "cpu"
        assert torch.equal(actual, expected.cpu())


def test_save_serializes_cpu_state_dict_tensors(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "artifact.pt"

    artifact.save(save_path)

    payload = torch.load(save_path, map_location="cpu", weights_only=True)

    assert isinstance(payload, Mapping)
    assert "model_state_dict" in payload
    assert isinstance(payload["model_state_dict"], Mapping)

    for tensor in payload["model_state_dict"].values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"


def test_load_raises_type_error_for_non_mapping_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad_payload.pt"
    path.write_bytes(b"x")

    def fake_load(*args, **kwargs):
        return ["not", "a", "mapping"]

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(TypeError, match="expected mapping payload"):
        GCNNArtifact.load(path)


def test_load_raises_value_error_when_required_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_key.pt"
    path.write_bytes(b"x")

    payload = {
        "model_state_dict": {},
        "hidden_channels": 5,
        "num_classes": 2,
        "dropout": 0.1,
        "alpha": 0.9,
        "K": 2,
        "k_neighbors": 4,
        "label_col": "label",
        "feature_cols": ("f1", "f2", "f3"),
        "region_col": "region",
        "pos_cols": ("x", "y"),
        # classes_ intentionally missing
    }

    def fake_load(*args, **kwargs):
        return payload

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(ValueError, match="missing key 'classes_'"):
        GCNNArtifact.load(path)


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

    loaded = GCNNArtifact.load(path)

    assert isinstance(loaded, GCNNArtifact)
    assert len(calls) == 2
    assert "weights_only" in calls[0]
    assert "weights_only" not in calls[1]