from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from core.latent_diffusion.data.builders import build_outpaint_dataset
from core.latent_diffusion.data.datasets import OutpaintDataset
from core.pixel_diffusion.precompute.base import PixelPrecomputeTask


type OutpaintBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class OutpaintPrecomputeTask(PixelPrecomputeTask):
    """
    Precompute strategy for outpainting pixel-diffusion training data.

    This strategy builds outpainting train/validation datasets, uses the masked
    image as the encoder input, uses the full image as the training target, and
    stores the per-sample bounding box as metadata.
    """

    masks_per_image_train: int = 5
    masks_per_image_val: int = 5
    img_size: int = 512

    def build_dataset(self, root_dir: Path) -> tuple[OutpaintDataset, OutpaintDataset]:
        return build_outpaint_dataset(
            root_dir=root_dir,
            masks_per_image_train=self.masks_per_image_train,
            masks_per_image_val=self.masks_per_image_val,
            img_size=self.img_size,
        )

    def get_encoder_input(self, batch: OutpaintBatch) -> torch.Tensor:
        masked_img, _, _ = batch
        return masked_img

    def get_target_img(self, batch: OutpaintBatch) -> torch.Tensor:
        _, img, _ = batch
        return img

    def get_sample_name(
        self,
        *,
        dataset: OutpaintDataset,
        batch: OutpaintBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> str:
        img_idx = global_idx // dataset.masks_per_image
        mask_idx = global_idx % dataset.masks_per_image
        base_name = Path(dataset.img_files[img_idx]).stem
        return f"{base_name}_k{mask_idx:03d}"

    def get_metadata(
        self,
        *,
        dataset: OutpaintDataset,
        batch: OutpaintBatch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> dict[str, torch.Tensor]:
        _, _, bbox = batch
        return {
            "bbox": bbox[batch_idx].detach().cpu(),
        }