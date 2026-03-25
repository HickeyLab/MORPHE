from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

import torch
from torch.utils.data import Dataset


TDataset = TypeVar("TDataset", bound=Dataset)
TBatch = TypeVar("TBatch")
TMetadata = TypeVar("TMetadata", bound=dict[str, object])
@dataclass(frozen=True)
class PixelPrecomputeStrategy(ABC, Generic[TDataset, TBatch, TMetadata]):
    """
    Abstract interface for pixel-diffusion dataset precomputation strategies.

    Implementations define how train/validation datasets are built, how encoder
    inputs and target images are extracted from each batch, how samples are
    named, and what metadata is written for each saved example.
    """
    
    ae_pretrained_path: str = "runwayml/stable-diffusion-v1-5"

    @abstractmethod
    def build_dataset(self, root_dir: Path) -> tuple[TDataset, TDataset]:
        """
        Build and return the train and validation datasets.
        """
        raise NotImplementedError

    @abstractmethod
    def get_encoder_input(self, batch: TBatch) -> torch.Tensor:
        """
        Extract the batched image tensor to feed into the VAE encoder.

        The returned tensor is typically shaped `[B, C, H, W]`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_target_img(self, batch: TBatch) -> torch.Tensor:
        """
        Extract the batched target image tensor to save for training.

        The returned tensor is typically shaped `[B, C, H, W]`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_name(
        self,
        *,
        dataset: TDataset,
        batch: TBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> str:
        """
        Return the filename stem, without the `.pt` suffix, for one sample.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metadata(
        self,
        *,
        dataset: TDataset,
        batch: TBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> TMetadata:
        """
        Return additional per-sample metadata to include in the saved `.pt` file.
        """
        raise NotImplementedError