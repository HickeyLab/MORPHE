from typing import Any, Sequence

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data, Dataset


class RegionGraphDataset(Dataset):
    """
    PyTorch Geometric dataset that groups rows by region and builds one
    k-nearest-neighbor graph per region.

    Each graph uses the rows within a single region as nodes, the selected
    feature columns as node features, and spatial nearest-neighbor
    relationships from ``pos_cols`` as edges. When ``label_col`` is provided,
    labels are encoded to integer class indices and attached to each graph's
    ``y`` tensor.

    Attributes:
        df: Copy of the input dataframe with a reset integer index.
        feature_cols: Node feature column names.
        pos_cols: Position column names used to construct the KNN graph.
        label_col: Optional label column name.
        region_col: Column that identifies which rows belong to the same region.
        k_neighbors: Number of nearest neighbors to connect for each node.
        classes_: Ordered class names used for label encoding, or ``None`` when
            labels are not used.
        region_ids: Unique region identifiers discovered in the dataframe.
        data_list: List of per-region PyG ``Data`` objects.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        label_col: str | None = "Cell Type",
        *,
        region_col: str = "region",
        pos_cols: Sequence[str] = ("x", "y"),
        k_neighbors: int = 20,
        classes_: Sequence[str] | None = None,
    ) -> None:
        """
        Initialize the dataset and precompute one graph per region.

        Args:
            df: Input dataframe containing features, coordinates, region IDs,
                and optionally labels.
            feature_cols: Column names to use as node features.
            label_col: Optional column containing node labels. If ``None``,
                labels are omitted and graphs are built for inference-only use.
            region_col: Column name that groups rows into regions, where each
                region becomes a separate graph.
            pos_cols: Coordinate column names used to compute nearest neighbors.
            k_neighbors: Number of nearest neighbors to connect per node.
            classes_: Optional fixed class ordering for label encoding. This is
                useful at inference time to ensure labels align with the
                training artifact.

        Raises:
            ValueError: If ``label_col`` contains labels not present in
                ``classes_`` when an explicit class list is provided.
        """
        super().__init__()

        # Reset index to ensure row_idx correctness
        self.df = df.reset_index(drop=True)

        # IMPORTANT: convert to list (not tuple!)
        self.feature_cols = list(feature_cols)
        self.pos_cols = list(pos_cols)

        self.label_col = label_col
        self.region_col = str(region_col)
        self.k_neighbors = int(k_neighbors)

        # Label encoding (training only)
        if self.label_col is not None:
            if classes_ is None:
                self.classes_ = sorted(
                    self.df[self.label_col].astype(str).unique().tolist()
                )
            else:
                self.classes_ = list(classes_)

            label_to_index = {c: i for i, c in enumerate(self.classes_)}
            self.df[self.label_col] = self.df[self.label_col].map(label_to_index)

            if self.df[self.label_col].isna().any():
                raise ValueError(
                    "label_col contains labels not present in classes_."
                )
        else:
            self.classes_ = None

        # Build region graphs
        self.region_ids = self.df[self.region_col].unique()
        self.data_list: list[Data] = []

        for region_id in self.region_ids:
            region_df = self.df[self.df[self.region_col] == region_id]

            features = region_df.loc[:, self.feature_cols].values
            positions = region_df.loc[:, self.pos_cols].values
            row_idx = region_df.index.to_numpy()

            num_nodes = features.shape[0]

            # Build KNN graph
            if num_nodes <= 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                knn = NearestNeighbors(
                    n_neighbors=min(self.k_neighbors + 1, num_nodes),
                    algorithm="ball_tree",
                )
                knn.fit(positions)

                _, indices = knn.kneighbors(positions)

                edge_list = []
                for i, nbrs in enumerate(indices):
                    for nbr in nbrs:
                        if nbr != i:
                            edge_list.append([i, nbr])

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                
                # TODO: .uniqe() blowing up memory. temporarily removing
                # Make graph undirected
                edge_index = torch.cat(
                    [edge_index, edge_index.flip(0)],
                    dim=1,
                )
                #.unique(dim=1)

            # Create PyG Data object
            x = torch.tensor(features, dtype=torch.float32)

            data = Data(
                x=x,
                edge_index=edge_index,
            )

            data.region_id = region_id
            data.row_idx = torch.tensor(row_idx, dtype=torch.long)

            # Attach labels only during training
            if self.label_col is not None:
                labels = region_df[self.label_col].values
                data.y = torch.tensor(labels, dtype=torch.long)

            self.data_list.append(data)

    def __len__(self) -> int:
        """
        Return the number of region graphs in the dataset.

        Returns:
            The number of precomputed ``Data`` objects, one per region.
        """
        return len(self.data_list)

    def __getitem__(self, idx: Any) -> Data:
        """
        Return a single region graph by index.

        Scalar ``torch.Tensor`` indices are accepted for compatibility with
        PyTorch-style indexing and converted to Python integers.

        Args:
            idx: Dataset index as an integer-like value or a scalar tensor.

        Returns:
            The region-level PyG ``Data`` object at the requested index.

        Raises:
            TypeError: If ``idx`` is a non-scalar tensor.
        """
        if isinstance(idx, torch.Tensor):
            if idx.ndim != 0:
                raise TypeError("RegionGraphDataset only supports scalar indexing.")
            idx = int(idx.item())
        else:
            idx = int(idx)

        return self.data_list[idx]