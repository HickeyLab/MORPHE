from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import LatentDiffusionTrainer

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset

from ....constants import InferenceMode
from ..architecture import LatentArchitectureSpec
from .config import LatentTrainerConfig


class LatentTrainTask(ABC):
    """
    Minimal plugin-style strategy.
    Each task implements its own train_step logic.
    """
    supports_decay: bool = False
    
    train_num_workers: int = 4
    val_num_workers: int = 2
    
    @property
    @abstractmethod
    def inference_mode(self) -> InferenceMode:
        raise NotImplementedError

    @abstractmethod
    def build_architecture_spec(self, cfg: LatentTrainerConfig,) -> LatentArchitectureSpec:
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self, input_dir: Path) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self,
        trainer: "LatentDiffusionTrainer",
        batch: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def validate_step(self, trainer: "LatentDiffusionTrainer") -> float:
        raise NotImplementedError