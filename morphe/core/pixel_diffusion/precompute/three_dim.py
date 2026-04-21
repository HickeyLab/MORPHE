from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from ...latent_diffusion.data.builders import build_three_dim_dataset
from ...latent_diffusion.data.datasets import Slice3DDataset
from .base import PixelPrecomputeTask


type ThreeDimBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class ThreeDimImputationPrecomputeTask(PixelPrecomputeTask):
    """
    Precompute strategy for 3D imputation pixel-diffusion training data.

    This strategy builds the train/validation 3D slice datasets, uses the
    previous slice image as the encoder input, uses the ground-truth slice as
    the target image, and stores the interpolation weights as per-sample
    metadata.
    """

    def build_dataset(self, input_dir: Path) -> tuple[Slice3DDataset, Slice3DDataset]:
        return build_three_dim_dataset(input_dir=input_dir)

    def get_encoder_input(self, batch: ThreeDimBatch) -> torch.Tensor:
        img_prev, _, _, _, _ = batch
        return img_prev

    def get_target_img(self, batch: ThreeDimBatch) -> torch.Tensor:
        _, _, img_gt, _, _ = batch
        return img_gt

    def get_sample_name(
        self,
        *,
        dataset: Slice3DDataset,
        batch: ThreeDimBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> str:
        return f"{split_name}_{global_idx:07d}"

    def get_metadata(
        self,
        *,
        dataset: Slice3DDataset,
        batch: ThreeDimBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> dict[str, float]:
        _, _, _, w_prev, w_next = batch
        return {
            "w_prev": float(w_prev[batch_idx].item()),
            "w_next": float(w_next[batch_idx].item()),
        }