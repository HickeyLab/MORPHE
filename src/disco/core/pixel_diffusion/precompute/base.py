from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

@dataclass
class PixelPrecomputeStrategy(ABC):
    @abstractmethod
    def build_dataset(self, root_dir: Path) -> tuple[Dataset, Dataset]:
        """
        Return the train and val datasets.
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_encoder_input(self, batch: Any) -> torch.Tensor:
        """
        Return the batched image tensor that should be fed into the VAE encoder.
        Shape is typically [B, C, H, W].
        """
        raise NotImplementedError

    @abstractmethod
    def get_target_img(self, batch: Any) -> torch.Tensor:
        """
        Return the batched ground-truth / training target image tensor to save.
        Shape is typically [B, C, H, W].
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_name(
        self,
        *,
        dataset: Any,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> str:
        """
        Return the filename stem (without .pt) for the sample.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metadata(
        self,
        *,
        dataset: Any,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> dict[str, Any]:
        """
        Return any extra metadata to save into the .pt file.
        """
        raise NotImplementedError