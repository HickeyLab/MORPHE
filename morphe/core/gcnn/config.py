from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class GCNNTrainerConfig:
    """
    Configuration for training the graph convolutional classifier (GCNN).

    This config controls:
    - how the input dataframe is interpreted as a graph
    - the architecture of the GCN model
    - optimization and training behavior

    Dataset / graph construction:
        label_col: Name of the column containing class labels.
        region_col: Column identifying distinct regions (each becomes a graph).
        pos_cols: Tuple of (x, y) coordinate columns used to build spatial edges.
        k_neighbors: Number of nearest neighbors used to construct graph edges.
        batch_size: Number of graphs per batch during training.

    Model hyperparameters:
        hidden_channels: Dimensionality of hidden GCN layers.
        dropout: Dropout probability applied in the model.
        alpha: Teleport probability / residual weighting factor in the GCN.
        K: Number of propagation steps (graph convolution depth).

    Optimization:
        lr: Learning rate for the optimizer.
        weight_decay: L2 regularization strength.
        epochs: Number of full passes over the dataset.

    Expected dataframe schema:
        The input dataframe must contain:
        - all feature columns passed to the trainer
        - label_col
        - region_col
        - both columns in pos_cols

    Notes:
        - Each unique value in `region_col` defines a separate graph.
        - Edges are constructed using k-nearest neighbors in (x, y) space.
        - Larger `k_neighbors` increases connectivity but also compute cost.
        - Increasing `hidden_channels` improves capacity but increases memory usage.
        - `K` controls how far information propagates across the graph.

    Typical usage:
        cfg = GCNNTrainerConfig(
            hidden_channels=512,
            k_neighbors=15,
            epochs=50,
        )
    """
    # dataset/schema
    label_col: str = "Cell Type"
    region_col: str = "unique_region"
    pos_cols: tuple[str, str] = ("x", "y")
    k_neighbors: int = 20
    batch_size: int = 1

    # model hyperparams
    hidden_channels: int = 768
    dropout: float = 0.1
    alpha: float = 0.9
    K: int = 20

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 40
    
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        # dataset/schema
        if not isinstance(self.label_col, str) or not self.label_col:
            raise TypeError("label_col must be a non-empty str.")
        if not isinstance(self.region_col, str) or not self.region_col:
            raise TypeError("region_col must be a non-empty str.")

        if (
            not isinstance(self.pos_cols, (tuple, list))
            or len(self.pos_cols) != 2
            or not all(isinstance(c, str) and c for c in self.pos_cols)
        ):
            raise TypeError("pos_cols must be a length-2 sequence of non-empty str.")

        if not isinstance(self.k_neighbors, int):
            raise TypeError("k_neighbors must be an int.")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1.")
        
        if not isinstance(self.batch_size, int):
            raise TypeError(f"batch_size must be an int, got {type(self.batch_size)}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        
        # dataset/schema
        if not isinstance(self.hidden_channels, int):
            raise TypeError("hidden_channels must be an int.")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1.")

        if not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a float in [0, 1).")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("dropout must be in [0, 1).")

        if not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be a float in (0, 1].")
        if not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError("alpha must be in (0, 1].")

        if not isinstance(self.K, int):
            raise TypeError("K must be an int.")
        if self.K < 1:
            raise ValueError("K must be >= 1.")

        # optimization
        if not isinstance(self.lr, (int, float)):
            raise TypeError("lr must be a float > 0.")
        if not (float(self.lr) > 0.0):
            raise ValueError("lr must be > 0.")

        if not isinstance(self.weight_decay, (int, float)):
            raise TypeError("weight_decay must be a float >= 0.")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be >= 0.")

        if not isinstance(self.epochs, int):
            raise TypeError("epochs must be an int.")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")