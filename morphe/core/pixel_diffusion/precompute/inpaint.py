from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ...latent_diffusion.data.builders import build_inpaint_dataset
from ...latent_diffusion.data.datasets import InpaintDataset
from .base import PixelPrecomputeTask


type InpaintBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class InpaintPrecomputeTask(PixelPrecomputeTask):
    """
    Precompute strategy for inpainting pixel-diffusion training data.

    This strategy builds inpainting train/validation datasets, uses the masked
    image as the encoder input, uses the full image as the training target, and
    stores the per-sample mask as metadata.
    """

    masks_per_image_train: int = 2
    masks_per_image_val: int = 5
    img_size: int = 512

    def build_dataset(self, root_dir: Path) -> tuple[InpaintDataset, InpaintDataset]:
        return build_inpaint_dataset(
            root_dir=root_dir,
            masks_per_image_train=self.masks_per_image_train,
            masks_per_image_val=self.masks_per_image_val,
            img_size=self.img_size,
        )

    def get_encoder_input(self, batch: InpaintBatch) -> torch.Tensor:
        masked_img, _, _ = batch
        return masked_img

    def get_target_img(self, batch: InpaintBatch) -> torch.Tensor:
        _, img, _ = batch
        return img

    def get_sample_name(
        self,
        *,
        dataset: InpaintDataset,
        batch: InpaintBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> str:
        return f"{split_name}_{global_idx:07d}"

    def get_metadata(
        self,
        *,
        dataset: InpaintDataset,
        batch: InpaintBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> dict[str, torch.Tensor]:
        _, _, mask = batch
        return {
            "mask": mask[batch_idx].detach().cpu(),
        }