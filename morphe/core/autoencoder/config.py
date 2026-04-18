from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
@dataclass(frozen=True, slots=True)
class AutoencoderTrainerConfig:
    """
    Configuration for training the autoencoder on probability-vector inputs.

    This config controls:
    - how input features are selected from the dataframe
    - the architecture of the autoencoder
    - training/validation splitting and batching
    - optimization and loss weighting

    Data / dataset:
        val_ratio: Fraction of data reserved for validation (0–1).
        batch_size: Number of samples per training batch.
        num_workers: Number of workers for DataLoader parallelism.
        input_cols: Columns used as input features. If None, columns prefixed with
            "prob_" are automatically selected.

    Model architecture:
        bottleneck_dim: Dimensionality of the latent space (z).
        hidden_dim: Size of hidden layers in the encoder/decoder.

    Optimization:
        num_epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        alpha: Weight applied to the contrastive (clustering) loss component.

    Expected dataframe schema:
        The input dataframe must contain:
        - probability-like columns (either explicitly provided via `input_cols`
          or inferred as columns starting with "prob_")
        - no null values in the selected input columns

    Notes:
        - Inputs are treated as probability distributions and trained using
          KL-divergence reconstruction loss.
        - A contrastive loss is applied in latent space to encourage biologically
          meaningful clustering.
        - Smaller `bottleneck_dim` forces stronger compression but may lose detail.
        - Larger `hidden_dim` increases model capacity and memory usage.
        - `alpha` balances reconstruction vs. clustering:
            - lower alpha → better reconstruction
            - higher alpha → stronger latent structure

    Typical usage:
        cfg = AutoencoderTrainerConfig(
            bottleneck_dim=3,
            hidden_dim=512,
            num_epochs=100,
            lr=1e-4,
        )
    """
    val_ratio: float = 0.1
    batch_size: int = 4096
    num_workers: int = 4
    
    input_cols: Sequence[str] | None = None

    bottleneck_dim: int = 3
    hidden_dim: int = 512

    num_epochs: int = 100
    lr: float = 1e-6
    alpha: float = 0.1
    
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        # val_ratio
        if not isinstance(self.val_ratio, (int, float)):
            raise TypeError("val_ratio must be a float in [0, 1).")
        if not (0.0 < float(self.val_ratio) < 1.0):
            raise ValueError("val_ratio must be in [0, 1).")

        # ints
        for name, val in (
            ("batch_size", self.batch_size),
            ("num_workers", self.num_workers),
            ("bottleneck_dim", self.bottleneck_dim),
            ("hidden_dim", self.hidden_dim),
            ("num_epochs", self.num_epochs),
        ):
            if not isinstance(val, int):
                raise TypeError(f"{name} must be an int.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be > 0.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0.")

        # lr
        if not isinstance(self.lr, (int, float)):
            raise TypeError("lr must be a float > 0.")
        if not (float(self.lr) > 0.0):
            raise ValueError("lr must be > 0.")

        # alpha
        if not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be a float >= 0.")
        if float(self.alpha) < 0.0:
            raise ValueError("alpha must be >= 0.")
        
        # input cols
        if self.input_cols is not None:
            if not isinstance(self.input_cols, Sequence):
                raise TypeError("input_cols must be a sequence of strings or None.")
            if len(self.input_cols) == 0:
                raise ValueError("input_cols cannot be empty if provided.")
            if not all(isinstance(col, str) for col in self.input_cols):
                raise TypeError("All input_cols entries must be strings.")