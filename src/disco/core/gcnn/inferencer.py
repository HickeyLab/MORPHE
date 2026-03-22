from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from disco.core.gcnn.artifact import GCNNArtifact
from disco.core.gcnn.data import RegionGraphDataset
from disco.core.gcnn.model import GCNClassifier

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
    ) -> None:
        """
        Initialize the inferencer.

        Args:
            artifact: Serialized artifact containing model metadata and weights.
            model: Reconstructed GCNN model ready for inference.
        """
        self.artifact = artifact
        self.model = model

    @classmethod
    def from_artifact(
        cls,
        artifact: GCNNArtifact,
        *,
        device: torch.device | str | None = None,
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
        model = artifact.build_model(device=GCNNInferencer.resolve_device(device))
        return cls(artifact=artifact, model=model)
    
    @staticmethod
    def resolve_device(device: torch.device | str | None) -> torch.device:
        """
        Resolve a device specification into a concrete ``torch.device``.

        Args:
            device: Requested device. If ``None``, CUDA is used when available;
                otherwise CPU is used.

        Returns:
            A normalized torch device.
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

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

        model_device = next(self.model.parameters()).device
        all_probs: list[np.ndarray] = []
        all_rows: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(model_device)
                out = self.model(batch.x, batch.edge_index)
                probs = torch.softmax(out, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_rows.append(batch.row_idx.cpu().numpy())

        probs_all = np.concatenate(all_probs, axis=0)
        rows_all = np.concatenate(all_rows, axis=0)

        x_col, y_col = self.artifact.pos_cols[0], self.artifact.pos_cols[1]
        region_col = self.artifact.region_col

        result_df = df.loc[rows_all, [x_col, y_col, region_col]].copy().reset_index(drop=True)

        for class_idx in range(probs_all.shape[1]):
            result_df[f"prob_class{class_idx}"] = probs_all[:, class_idx]

        ordered_cols = [x_col, y_col, region_col] + [
            f"prob_class{class_idx}" for class_idx in range(probs_all.shape[1])
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