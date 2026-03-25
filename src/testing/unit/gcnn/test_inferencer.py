from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.gcnn.inferencer import GCNNInferencer


class DummyArtifactBase:
    """Base class used so isinstance checks can be patched cleanly."""


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Return fixed logits for 2 classes.
        # Shape: [num_rows_in_batch, num_classes]
        n = x.shape[0]
        logits = torch.tensor([[2.0, 1.0]], dtype=torch.float32).repeat(n, 1)
        return logits.to(x.device)


class DummyArtifact(DummyArtifactBase):
    def __init__(self) -> None:
        self.feature_cols = ["f1", "f2"]
        self.region_col = "region"
        self.pos_cols = ("x", "y")
        self.k_neighbors = 3
        self.classes_ = ["A", "B"]
        self.build_model_calls: list[tuple[torch.device, torch.dtype]] = []

    def build_model(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> DummyModel:
        self.build_model_calls.append((device, dtype))
        model = DummyModel().to(device=device, dtype=dtype)
        return model


class DummyBatch:
    def __init__(self, row_idx: list[int]) -> None:
        self.row_idx = torch.tensor(row_idx, dtype=torch.long)
        self.x = torch.ones((len(row_idx), 2), dtype=torch.float32)
        self.edge_index = torch.zeros((2, 0), dtype=torch.long)

    def to(self, device: torch.device) -> "DummyBatch":
        self.row_idx = self.row_idx.to(device)
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        return self


class DummyRegionGraphDataset:
    last_init_kwargs: dict | None = None

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str | None,
        region_col: str,
        pos_cols: tuple[str, str],
        k_neighbors: int,
    ) -> None:
        DummyRegionGraphDataset.last_init_kwargs = {
            "df": df,
            "feature_cols": feature_cols,
            "label_col": label_col,
            "region_col": region_col,
            "pos_cols": pos_cols,
            "k_neighbors": k_neighbors,
        }
        self.df = df

    def __len__(self) -> int:
        return len(self.df)


class DummyDataLoader:
    last_init_kwargs: dict | None = None

    def __init__(
        self,
        dataset: DummyRegionGraphDataset,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        DummyDataLoader.last_init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
        }
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.dataset.df)
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            yield DummyBatch(list(range(start, stop)))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [10, 20, 30],
            "y": [1, 2, 3],
            "region": ["r1", "r2", "r3"],
            "f1": [0.1, 0.2, 0.3],
            "f2": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def artifact() -> DummyArtifact:
    return DummyArtifact()


@pytest.fixture
def inferencer(artifact: DummyArtifact) -> GCNNInferencer:
    return GCNNInferencer(artifact=artifact, model=DummyModel())


def test_init_sets_fields(artifact: DummyArtifact) -> None:
    model = DummyModel()

    inferencer = GCNNInferencer(artifact=artifact, model=model)

    assert inferencer.artifact is artifact
    assert inferencer.model is model


def test_from_artifact_builds_model_with_resolved_device_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
    artifact: DummyArtifact,
) -> None:
    target_device = torch.device("cpu")
    target_dtype = torch.float32

    monkeypatch.setattr("core.gcnn.inferencer.GCNNArtifact", DummyArtifactBase)
    monkeypatch.setattr(
        "core.gcnn.inferencer.resolve_device",
        lambda device: target_device,
    )
    monkeypatch.setattr(
        "core.gcnn.inferencer.resolve_dtype",
        lambda device, dtype: target_dtype,
    )

    inferencer = GCNNInferencer.from_artifact(
        artifact,
        device="cpu",
        dtype=torch.float16,
    )

    assert isinstance(inferencer, GCNNInferencer)
    assert isinstance(inferencer.model, DummyModel)
    assert inferencer.artifact is artifact
    assert artifact.build_model_calls == [(target_device, target_dtype)]


def test_from_artifact_raises_for_wrong_artifact_type() -> None:
    with pytest.raises(TypeError, match="Expected artifact to be an instance of GCNNArtifact"):
        GCNNInferencer.from_artifact(object())  # type: ignore[arg-type]


def test_predict_proba_returns_expected_dataframe(
    monkeypatch: pytest.MonkeyPatch,
    inferencer: GCNNInferencer,
    sample_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr("core.gcnn.inferencer.RegionGraphDataset", DummyRegionGraphDataset)
    monkeypatch.setattr("core.gcnn.inferencer.DataLoader", DummyDataLoader)

    result = inferencer.predict_proba(sample_df, batch_size=2, shuffle=False)

    expected_probs = torch.softmax(torch.tensor([2.0, 1.0]), dim=0).numpy()

    assert list(result.columns) == ["x", "y", "region", "prob_A", "prob_B"]
    assert result["x"].tolist() == [10, 20, 30]
    assert result["y"].tolist() == [1, 2, 3]
    assert result["region"].tolist() == ["r1", "r2", "r3"]
    assert np.allclose(result["prob_A"].to_numpy(), [expected_probs[0]] * 3)
    assert np.allclose(result["prob_B"].to_numpy(), [expected_probs[1]] * 3)

    assert DummyRegionGraphDataset.last_init_kwargs is not None
    assert DummyRegionGraphDataset.last_init_kwargs["feature_cols"] == ["f1", "f2"]
    assert DummyRegionGraphDataset.last_init_kwargs["label_col"] is None
    assert DummyRegionGraphDataset.last_init_kwargs["region_col"] == "region"
    assert DummyRegionGraphDataset.last_init_kwargs["pos_cols"] == ("x", "y")
    assert DummyRegionGraphDataset.last_init_kwargs["k_neighbors"] == 3

    assert DummyDataLoader.last_init_kwargs is not None
    assert DummyDataLoader.last_init_kwargs["batch_size"] == 2
    assert DummyDataLoader.last_init_kwargs["shuffle"] is False


def test_predict_proba_writes_csv(
    monkeypatch: pytest.MonkeyPatch,
    inferencer: GCNNInferencer,
    sample_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("core.gcnn.inferencer.RegionGraphDataset", DummyRegionGraphDataset)
    monkeypatch.setattr("core.gcnn.inferencer.DataLoader", DummyDataLoader)

    output_path = tmp_path / "predictions" / "out.csv"
    result = inferencer.predict_proba(
        sample_df,
        batch_size=2,
        shuffle=False,
        output_file_path=output_path,
    )

    assert output_path.exists()

    loaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(loaded, result, check_dtype=False)


def test_predict_proba_respects_row_order_from_batches(
    monkeypatch: pytest.MonkeyPatch,
    inferencer: GCNNInferencer,
    sample_df: pd.DataFrame,
) -> None:
    class ReorderedLoader:
        def __init__(self, dataset, *, batch_size: int, shuffle: bool) -> None:
            self.dataset = dataset

        def __iter__(self):
            yield DummyBatch([2, 0])
            yield DummyBatch([1])

    monkeypatch.setattr("core.gcnn.inferencer.RegionGraphDataset", DummyRegionGraphDataset)
    monkeypatch.setattr("core.gcnn.inferencer.DataLoader", ReorderedLoader)

    result = inferencer.predict_proba(sample_df)

    assert result["x"].tolist() == [30, 10, 20]
    assert result["region"].tolist() == ["r3", "r1", "r2"]


def test_validate_inputs_raises_when_df_not_dataframe(
    inferencer: GCNNInferencer,
) -> None:
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        inferencer._validate_inputs(  # pyright: ignore[reportArgumentType]
            df="not a dataframe",
            batch_size=1,
            output_file_path=None,
        )


def test_validate_inputs_raises_when_df_empty(
    inferencer: GCNNInferencer,
) -> None:
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="df must not be empty"):
        inferencer._validate_inputs(
            df=empty_df,
            batch_size=1,
            output_file_path=None,
        )


@pytest.mark.parametrize("bad_batch_size", [True, False, 1.5, "2", None])
def test_validate_inputs_raises_when_batch_size_wrong_type(
    inferencer: GCNNInferencer,
    sample_df: pd.DataFrame,
    bad_batch_size,
) -> None:
    with pytest.raises(TypeError, match="batch_size must be an int"):
        inferencer._validate_inputs(
            df=sample_df,
            batch_size=bad_batch_size,  # type: ignore[arg-type]
            output_file_path=None,
        )


@pytest.mark.parametrize("bad_batch_size", [0, -1, -5])
def test_validate_inputs_raises_when_batch_size_nonpositive(
    inferencer: GCNNInferencer,
    sample_df: pd.DataFrame,
    bad_batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        inferencer._validate_inputs(
            df=sample_df,
            batch_size=bad_batch_size,
            output_file_path=None,
        )


def test_validate_inputs_raises_for_non_csv_output_path(
    inferencer: GCNNInferencer,
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="output_file_path must have a .csv extension"):
        inferencer._validate_inputs(
            df=sample_df,
            batch_size=1,
            output_file_path=Path("predictions.txt"),
        )


def test_validate_inputs_raises_when_pos_cols_too_short(
    inferencer: GCNNInferencer,
    sample_df: pd.DataFrame,
    artifact: DummyArtifact,
) -> None:
    artifact.pos_cols = ("x",)

    with pytest.raises(ValueError, match="artifact.pos_cols must contain at least two columns"):
        inferencer._validate_inputs(
            df=sample_df,
            batch_size=1,
            output_file_path=None,
        )