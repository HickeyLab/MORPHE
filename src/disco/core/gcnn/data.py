from typing import Sequence
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Dataset, Data
import torch

class RegionGraphDataset(Dataset):
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        label_col: str | None = "Cell Type",
        *,
        region_col: str = "unique_region",
        pos_cols: Sequence[str] = ("x", "y"),
        k_neighbors: int = 20,
        classes_: Sequence[str] | None = None,
    ):
        super().__init__()
        
        # Validate inputs first
        self._validate_inputs(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            region_col=region_col,
            pos_cols=pos_cols,
            k_neighbors=k_neighbors
        )
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.region_col = str(region_col)
        self.pos_cols = tuple(pos_cols)
        self.k_neighbors = int(k_neighbors)

        # Only get unique class to index mapping for training
        if self.label_col is not None:
            if classes_ is None:
                self.classes_ = sorted(self.df[self.label_col].astype(str).unique().tolist())
            else:
                self.classes_ = list(classes_)
                
            # Label encoding
            label_to_index = {c: i for i, c in enumerate(self.classes_)}
            self.df[self.label_col] = self.df[self.label_col].map(label_to_index)
            
            if self.df[self.label_col].isna().any():
                raise ValueError("label_col contains labels not present in classes_.")
        else:
            self.classes_ = None

        self.region_ids = self.df[self.region_col].unique()
        self.data_list = []
        for region_id in self.region_ids:
            region_df = self.df[self.df[self.region_col] == region_id]

            features = region_df[self.feature_cols].values
            positions = region_df[self.pos_cols].values
            
            row_idx = region_df.index.to_numpy()

            num_nodes = features.shape[0]

            if num_nodes <= 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            else:
                # ---------- KNN graph ----------
                knn = NearestNeighbors(
                    n_neighbors=min(self.k_neighbors + 1, num_nodes),
                    algorithm='ball_tree'
                )
                knn.fit(positions)

                distances, indices = knn.kneighbors(positions)

                # Build directed edges
                edge_list = []
                for i, nbrs in enumerate(indices):
                    for nbr in nbrs:
                        if nbr != i:
                            edge_list.append([i, nbr])

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

                # ---------- Make graph undirected (symmetric KNN) ----------
                edge_index = torch.cat(
                    [edge_index, edge_index.flip(0)],
                    dim=1
                ).unique(dim=1)

            # Create Data object
            x = torch.tensor(features, dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index)
            data.region_id = region_id
            data.row_idx = torch.tensor(row_idx, dtype=torch.long)
            
            # Only attach y to data during training, skip during inference
            if self.label_col is not None:
                labels = region_df[self.label_col].values
                y = torch.tensor(labels, dtype=torch.long)
                data.y = y

            self.data_list.append(data)
            
    def _validate_inputs(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        label_col: str | None,
        region_col: str,
        pos_cols: Sequence[str],
        k_neighbors: int,
    ) -> None:
        # Basic type validation
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df is empty.")
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty.")
        if len(set(feature_cols)) != len(feature_cols):
            raise ValueError("feature_cols contains duplicates.")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1.")

        # Check for required columns
        required = [region_col, *pos_cols, *feature_cols]
        
        # Only add label column to required columns for training, not inference
        if label_col is not None:
            required.append(label_col)
            
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {sorted(set(missing))}")

        # NaNs
        check_cols = [region_col, *pos_cols, *feature_cols]
        if df[check_cols].isna().any().any():
            raise ValueError("region/pos/feature columns contain NaNs.")
        if label_col is not None and df[label_col].isna().any():
            raise ValueError(f"label_col '{label_col}' contains NaNs.")

        # Numeric types
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in pos_cols):
            raise TypeError("pos_cols must be numeric.")
        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(f"Non-numeric feature columns: {non_numeric}")