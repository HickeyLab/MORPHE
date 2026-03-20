from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

import torch
from torch.utils.data import Dataset


if TYPE_CHECKING:
    from disco.core.latent_diffusion.train.diffusion_trainer import DiffusionTrainer


class LatentTrainStrategy(ABC):
    """
    Minimal plugin-style strategy.
    Each task implements its own train_step logic.
    """

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------
    requires_coord_encoder: bool = False
    requires_bbox_encoder: bool = False
    three_dimensional_cond_encoder: bool = False

    train_num_workers: int = 4
    val_num_workers: int = 2
    
    supports_decay: bool = False
    decay_enabled: bool = False
    
    patience: Optional[int] = None
    lr_decay_every: Optional[int] = None
    lr_decay_factor: Optional[float] = None

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    @abstractmethod
    def build_dataset(self, root_dir: Path) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    # -----------------------------------------------------
    # Training step (FULLY IMPLEMENTED BY STRATEGY)
    # -----------------------------------------------------
    @abstractmethod
    def train_step(
        self,
        trainer: "DiffusionTrainer",
        batch: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    # -----------------------------------------------------
    # Validation step
    # -----------------------------------------------------
    @abstractmethod
    def validate_step(self, trainer: "DiffusionTrainer") -> float:
        raise NotImplementedError