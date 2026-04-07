"""Smoke tests: verify artifact save/load round-trips."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.autoencoder.artifact import AutoencoderArtifact
from core.autoencoder.model import Autoencoder
from core.gcnn.artifact import GCNNArtifact
from core.gcnn.model import GCNClassifier
from core.pipeline.artifact import MorpheArtifact


# ── helpers ─────────────────────────────────────────────────────────


def _make_autoencoder_artifact(
    in_dim: int = 10,
    bottleneck_dim: int = 3,
    hidden_dim: int = 64,
) -> AutoencoderArtifact:
    model = Autoencoder(in_dim=in_dim, bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim)
    return AutoencoderArtifact(
        state_dict=model.state_dict(),
        input_cols=tuple(f"f{i}" for i in range(in_dim)),
        in_dim=in_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        z_min=torch.zeros(bottleneck_dim),
        z_max=torch.ones(bottleneck_dim),
    )


def _make_gcnn_artifact(
    in_channels: int = 4,
    hidden_channels: int = 16,
    num_classes: int = 3,
) -> GCNNArtifact:
    model = GCNClassifier(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        K=5,
    )
    return GCNNArtifact(
        model_state_dict=model.state_dict(),
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=0.1,
        alpha=0.9,
        K=5,
        k_neighbors=10,
        label_col="cell_type",
        feature_cols=tuple(f"f{i}" for i in range(in_channels)),
        region_col="region",
        pos_cols=("x", "y"),
        classes_=tuple(f"class_{i}" for i in range(num_classes)),
    )


# ── AutoencoderArtifact ───────────────────────────────────────────


class TestAutoencoderArtifactSmoke:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        original = _make_autoencoder_artifact()
        path = tmp_path / "ae.pt"

        original.save(path)
        loaded = AutoencoderArtifact.load(path)

        assert loaded.in_dim == original.in_dim
        assert loaded.bottleneck_dim == original.bottleneck_dim
        assert loaded.hidden_dim == original.hidden_dim
        assert loaded.input_cols == original.input_cols

    def test_build_model_produces_working_model(self) -> None:
        artifact = _make_autoencoder_artifact()
        model = artifact.build_model(device="cpu")

        x = torch.randn(2, 10)
        z, x_hat = model(x)

        assert z.shape == (2, 3)
        assert x_hat.shape == (2, 10)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        artifact = _make_autoencoder_artifact()
        path = tmp_path / "nested" / "deep" / "ae.pt"

        artifact.save(path)
        assert path.exists()


# ── GCNNArtifact ──────────────────────────────────────────────────


class TestGCNNArtifactSmoke:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        original = _make_gcnn_artifact()
        path = tmp_path / "gcnn.pt"

        original.save(path)
        loaded = GCNNArtifact.load(path)

        assert loaded.hidden_channels == original.hidden_channels
        assert loaded.num_classes == original.num_classes
        assert loaded.feature_cols == original.feature_cols
        assert loaded.classes_ == original.classes_

    def test_build_model_produces_working_model(self) -> None:
        artifact = _make_gcnn_artifact()
        model = artifact.build_model(device="cpu")

        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        logits = model(x, edge_index)

        assert logits.shape == (5, 3)

    def test_load_rejects_missing_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"model_state_dict": {}}, path)

        with pytest.raises(ValueError, match="missing key"):
            GCNNArtifact.load(path)


# ── MorpheArtifact ────────────────────────────────────────────────


class TestMorpheArtifactSmoke:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        original = MorpheArtifact()
        path = tmp_path / "morphe.pt"

        original.save(path)
        loaded = MorpheArtifact.load(path)

        assert loaded.preprocessor_config is None
        assert loaded.autoencoder_artifact is None
        assert loaded.gcnn_artifact is None

    def test_load_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MorpheArtifact.load(tmp_path / "nonexistent.pt")

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        artifact = MorpheArtifact()
        path = tmp_path / "a" / "b" / "morphe.pt"

        artifact.save(path)
        assert path.exists()
