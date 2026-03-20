from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class AutoencoderTrainerConfig:
    save_dir: Path = Path("checkpoints")
    save_best_only: bool = True
    val_ratio: float = 0.1
    batch_size: int = 4096
    num_workers: int = 4
    
    input_cols: list[str] | None = None

    bottleneck_dim: int = 3
    hidden_dim: int = 512

    num_epochs: int = 100
    lr: float = 1e-6
    alpha: float = 0.1

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
            if not isinstance(self.input_cols, list):
                raise TypeError("input_cols must be a list of strings or None.")
            if len(self.input_cols) == 0:
                raise ValueError("input_cols cannot be empty if provided.")
            if not all(isinstance(col, str) for col in self.input_cols):
                raise TypeError("All input_cols entries must be strings.")