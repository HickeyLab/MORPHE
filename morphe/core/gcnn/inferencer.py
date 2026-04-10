from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .artifact import GCNNArtifact
from .data import RegionGraphDataset
from .model import GCNClassifier
from ...utils import resolve_device, resolve_dtype

class GCNNInferencer:
    """
    Inference wrapper for a trained GCNN artifact.

    This class reconstructs the trained model from a ``GCNNArtifact`` and
    provides utilities for running batched probability prediction on a
    dataframe of input samples.
    """

    def __init__(
        self,
        *,
        artifact: GCNNArtifact,
        model: GCNClassifier,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize the inferencer.

        Args:
            artifact: Serialized artifact containing model metadata and weights.
            model: Reconstructed GCNN model ready for inference.
        """
        self.artifact = artifact
        self.model = model
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)

    @classmethod
    def from_artifact(
        cls,
        artifact: GCNNArtifact,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> GCNNInferencer:
        """
        Build an inferencer from a serialized GCNN artifact.

        Args:
            artifact: Serialized artifact containing model metadata and weights.
            device: Target device for inference. If omitted, CUDA is used when
                available; otherwise CPU is used.

        Returns:
            A ready-to-use ``GCNNInferencer``.
        """
        if not isinstance(artifact, GCNNArtifact):
            raise TypeError("Expected artifact to be an instance of GCNNArtifact.")
        
        device = resolve_device(device)
        dtype = resolve_dtype(device, dtype)
        model = artifact.build_model(device=device, dtype=dtype)
        
        return cls(artifact=artifact, model=model, device=device, dtype=dtype)

    def predict_proba(
        self,
        df: pd.DataFrame,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        output_file_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Predict per-class probabilities for each row in the input dataframe.

        Args:
            df: Input dataframe containing feature and positional columns.
            batch_size: Number of region graphs per batch.
            shuffle: Whether to shuffle the dataset during inference.
            output_file_path: Optional CSV path. When provided, predictions are
                also written to disk.

        Returns:
            A dataframe containing positional metadata and class-probability
            columns.
        """
        output_path = None if output_file_path is None else Path(output_file_path)
        self._validate_inputs(
            df=df,
            batch_size=batch_size,
            output_file_path=output_path,
        )

        dataset = RegionGraphDataset(
            df=df,
            feature_cols=self.artifact.feature_cols,
            label_col=None,
            region_col=self.artifact.region_col,
            pos_cols=self.artifact.pos_cols,
            k_neighbors=self.artifact.k_neighbors,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        all_probs: list[np.ndarray] = []
        all_rows: list[np.ndarray] = []
                
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                batch.x = batch.x.to(dtype=self.dtype)

                out = self.model(batch.x, batch.edge_index)
                probs = torch.softmax(out, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_rows.append(batch.row_idx.cpu().numpy())

        probs_all = np.concatenate(all_probs, axis=0)
        rows_all = np.concatenate(all_rows, axis=0)

        x_col, y_col = self.artifact.pos_cols[0], self.artifact.pos_cols[1]
        region_col = self.artifact.region_col

        result_df = df.loc[rows_all, [x_col, y_col, region_col]].copy().reset_index(drop=True)

        for class_idx, class_name in enumerate(self.artifact.classes_):
            result_df[f"prob_{class_name}"] = probs_all[:, class_idx]

        ordered_cols = [x_col, y_col, region_col] + [
            f"prob_{class_name}" for class_name in self.artifact.classes_
        ]
        result_df = result_df[ordered_cols]

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)

        return result_df

    def _validate_inputs(
        self,
        *,
        df: pd.DataFrame,
        batch_size: int,
        output_file_path: Path | None,
    ) -> None:
        """
        Validate user inputs for probability prediction.

        Args:
            df: Input dataframe.
            batch_size: Batch size for inference.
            output_file_path: Optional output CSV path.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If input values are invalid.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("df must not be empty.")

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be an int, got {type(batch_size)}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        if output_file_path is not None and output_file_path.suffix.lower() != ".csv":
            raise ValueError("output_file_path must have a .csv extension.")

        if len(self.artifact.pos_cols) < 2:
            raise ValueError("artifact.pos_cols must contain at least two columns.")