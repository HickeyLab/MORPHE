from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from src.disco.core.gcnn.artifact import GCNNArtifact
from src.disco.core.gcnn.data import RegionGraphDataset
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

@dataclass(frozen=True)
class GCNNInferencer:
    artifact: "GCNNArtifact"
    device: torch.device | str | None = None
    batch_size: int = 1
    shuffle: bool = False

    def predict_proba(
        self,
        df: pd.DataFrame,
        *,
        output_file_path: Path | None = None,
        save: bool = False,
    ) -> pd.DataFrame:
        self._validate_inputs(df=df, output_file_path=output_file_path, save=save)

        # Load and build model
        model = self.artifact.build_model(device=self.device)
        model_device = next(model.parameters()).device

        dataset = RegionGraphDataset(
            df=df,
            feature_cols=self.artifact.feature_cols,
            label_col=None,
            region_col=self.artifact.region_col,
            pos_cols=self.artifact.pos_cols,
            k_neighbors=self.artifact.k_neighbors,
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        model.eval()
        all_probs: list[np.ndarray] = []
        all_rows: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                # Each batch is ONE region graph (given batch_size=1)
                batch = batch.to(model_device)

                # GNN forward pass
                out = model(batch.x, batch.edge_index)  # [num_nodes, num_classes]

                # softmax probability for each node
                probs = torch.softmax(out, dim=1).cpu().numpy()

                all_probs.append(probs)
                all_rows.append(batch.row_idx.cpu().numpy())

        # Concatenate results across all regions
        probs_all = np.concatenate(all_probs, axis=0)  # [num_cells_total, num_classes]
        rows_all = np.concatenate(all_rows, axis=0)

        x_col, y_col = self.artifact.pos_cols[0], self.artifact.pos_cols[1]
        r_col = self.artifact.region_col

        result_df = df.loc[rows_all, [x_col, y_col, r_col]].copy().reset_index(drop=True)

        # Add prob columns
        for i in range(probs_all.shape[1]):
            result_df[f"prob_class{i}"] = probs_all[:, i]

        # Order columns
        cols = [x_col, y_col, r_col] + [f"prob_class{i}" for i in range(probs_all.shape[1])]
        result_df = result_df[cols]

        if save:
            result_df.to_csv(output_file_path, index=False)

        return result_df

    def _validate_inputs(
        self,
        *,
        df: pd.DataFrame,
        output_file_path: Optional[Path],
        save: bool,
    ) -> None:
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df is empty.")
        if save and not output_file_path:
            raise ValueError("Output file path empty.")
    