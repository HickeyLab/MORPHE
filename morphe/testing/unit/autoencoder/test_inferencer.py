from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.autoencoder.inferencer import AutoencoderInferencer


class DummyModel(torch.nn.Module):
    """Minimal model that acts like an Autoencoder for testing."""

    def __init__(self, in_dim: int = 5, bottleneck_dim: int = 3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.encoder = torch.nn.Linear(in_dim, bottleneck_dim, bias=False)
        self.decoder = torch.nn.Linear(bottleneck_dim, in_dim, bias=False)

        # Set deterministic weights for predictable output
        torch.nn.init.ones_(self.encoder.weight)
        torch.nn.init.ones_(self.decoder.weight)


class DummyArtifact:
    def __init__(
        self,
        *,
        in_dim: int = 5,
        bottleneck_dim: int = 3,
        input_cols: tuple[str, ...] = ("f1", "f2", "f3", "f4", "f5"),
    ) -> None:
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.input_cols = input_cols
        self.z_min = torch.tensor([0.0, 0.0, 0.0])
        self.z_max = torch.tensor([1.0, 1.0, 1.0])
        self.build_model_calls: list[tuple[torch.device, torch.dtype]] = []

    def build_model(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DummyModel:
        self.build_model_calls.append((device, dtype))
        model = DummyModel(in_dim=self.in_dim, bottleneck_dim=self.bottleneck_dim)
        return model.to(device=device, dtype=dtype)


@pytest.fixture
def artifact() -> DummyArtifact:
    return DummyArtifact()


@pytest.fixture
def inferencer(artifact: DummyArtifact) -> AutoencoderInferencer:
    model = DummyModel()
    inf = AutoencoderInferencer.__new__(AutoencoderInferencer)
    inf.device = torch.device("cpu")
    inf.dtype = torch.float32
    inf.artifact = artifact
    inf.model = model
    return inf


def test_from_artifact_raises_for_wrong_artifact_type() -> None:
    with pytest.raises(TypeError, match="Expected artifact to be an instance of AutoencoderArtifact"):
        AutoencoderInferencer.from_artifact(object())  # type: ignore[arg-type]


def test_to_embedding_tensor_converts_numpy(inferencer: AutoencoderInferencer) -> None:
    arr = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = inferencer._to_embedding_tensor(arr)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 5)


def test_to_embedding_tensor_passes_through_tensor(inferencer: AutoencoderInferencer) -> None:
    t = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = inferencer._to_embedding_tensor(t)
    assert result is t


def test_to_embedding_tensor_raises_for_invalid_type(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(TypeError, match="emb_matrix must be a torch.Tensor or numpy.ndarray"):
        inferencer._to_embedding_tensor([1.0, 2.0])  # type: ignore[arg-type]


def test_validate_embedding_matrix_accepts_valid_shape(inferencer: AutoencoderInferencer) -> None:
    t = torch.randn(4, 5)
    inferencer._validate_embedding_matrix(t)  # should not raise


def test_validate_embedding_matrix_rejects_wrong_ndim(inferencer: AutoencoderInferencer) -> None:
    t = torch.randn(5)
    with pytest.raises(ValueError, match="must have shape"):
        inferencer._validate_embedding_matrix(t)


def test_validate_embedding_matrix_rejects_wrong_feature_dim(inferencer: AutoencoderInferencer) -> None:
    t = torch.randn(4, 3)
    with pytest.raises(ValueError, match="must have 5 features"):
        inferencer._validate_embedding_matrix(t)


def test_validate_dataframe_columns_raises_on_missing_cols(inferencer: AutoencoderInferencer) -> None:
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    with pytest.raises(ValueError, match="missing required input columns"):
        inferencer._validate_dataframe_columns(df)


def test_validate_dataframe_columns_passes_on_valid_df(inferencer: AutoencoderInferencer) -> None:
    df = pd.DataFrame({f"f{i}": [float(i)] for i in range(1, 6)})
    inferencer._validate_dataframe_columns(df)  # should not raise


def test_validate_rgb_output_column_names_rejects_duplicates(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        inferencer._validate_rgb_output_column_names(r_col="R", g_col="R", b_col="B")


def test_validate_rgb_tensor_rejects_non_tensor(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(TypeError, match="rgb must be a torch.Tensor"):
        inferencer._validate_rgb_tensor(np.array([1, 2, 3]))  # type: ignore[arg-type]


def test_validate_rgb_tensor_rejects_wrong_ndim(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(ValueError, match="must have ndim 2 or 3"):
        inferencer._validate_rgb_tensor(torch.randn(5))


def test_validate_rgb_tensor_rejects_2d_wrong_channels(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(ValueError, match="must have shape \\(n, 3\\)"):
        inferencer._validate_rgb_tensor(torch.randn(4, 2))


def test_validate_rgb_tensor_rejects_3d_wrong_channels(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(ValueError, match="must have shape \\(3, h, w\\)"):
        inferencer._validate_rgb_tensor(torch.randn(4, 8, 8))


def test_validate_rgb_tensor_accepts_valid_2d(inferencer: AutoencoderInferencer) -> None:
    inferencer._validate_rgb_tensor(torch.randn(4, 3))  # should not raise


def test_validate_rgb_tensor_accepts_valid_3d(inferencer: AutoencoderInferencer) -> None:
    inferencer._validate_rgb_tensor(torch.randn(3, 8, 8))  # should not raise


def test_validate_rgb_image_rejects_non_tensor(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(TypeError, match="rgb must be a torch.Tensor"):
        inferencer._validate_rgb_image(np.array([1, 2, 3]))  # type: ignore[arg-type]


def test_validate_rgb_image_rejects_wrong_shape(inferencer: AutoencoderInferencer) -> None:
    with pytest.raises(ValueError, match="must have shape \\(3, h, w\\)"):
        inferencer._validate_rgb_image(torch.randn(4, 8, 8))


def test_validate_rgb_image_accepts_valid(inferencer: AutoencoderInferencer) -> None:
    inferencer._validate_rgb_image(torch.randn(3, 8, 8))  # should not raise


def test_encode_to_rgb_returns_uint8_tensor(inferencer: AutoencoderInferencer) -> None:
    emb = torch.randn(4, 5)
    result = inferencer.encode_to_rgb(emb)

    assert result.dtype == torch.uint8
    assert result.shape == (4, 3)
    assert result.device.type == "cpu"


def test_encode_to_rgb_df_adds_rgb_columns(inferencer: AutoencoderInferencer) -> None:
    df = pd.DataFrame({f"f{i}": np.random.randn(3) for i in range(1, 6)})

    result = inferencer.encode_to_rgb_df(df)

    assert "R" in result.columns
    assert "G" in result.columns
    assert "B" in result.columns
    assert len(result) == 3
    # Original columns preserved
    for col in df.columns:
        assert col in result.columns


def test_encode_to_rgb_df_custom_column_names(inferencer: AutoencoderInferencer) -> None:
    df = pd.DataFrame({f"f{i}": np.random.randn(2) for i in range(1, 6)})

    result = inferencer.encode_to_rgb_df(df, r_col="red", g_col="green", b_col="blue")

    assert "red" in result.columns
    assert "green" in result.columns
    assert "blue" in result.columns


def test_encode_to_rgb_df_does_not_mutate_input(inferencer: AutoencoderInferencer) -> None:
    df = pd.DataFrame({f"f{i}": np.random.randn(3) for i in range(1, 6)})
    original_cols = list(df.columns)

    inferencer.encode_to_rgb_df(df)

    assert list(df.columns) == original_cols


def test_decode_rgb_to_logits_2d(inferencer: AutoencoderInferencer) -> None:
    rgb = torch.rand(4, 3)
    result = inferencer.decode_rgb_to_logits(rgb)

    assert result.shape == (4, 5)


def test_decode_rgb_to_logits_3d(inferencer: AutoencoderInferencer) -> None:
    rgb = torch.rand(3, 4, 4)
    result = inferencer.decode_rgb_to_logits(rgb)

    assert result.shape == (4, 4, 5)


def test_decode_rgb_to_cell_map_shape(inferencer: AutoencoderInferencer) -> None:
    rgb = torch.rand(3, 4, 4)
    result = inferencer.decode_rgb_to_cell_map(rgb)

    assert result.shape == (1, 4, 4)
    assert result.dtype == torch.long


def test_decode_rgb_to_cell_map_white_pixels_are_zero(inferencer: AutoencoderInferencer) -> None:
    rgb = torch.ones(3, 2, 2)  # all white
    result = inferencer.decode_rgb_to_cell_map(rgb)

    assert (result == 0).all()
