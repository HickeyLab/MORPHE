from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from typing import Any

import torch

from disco.core.latent_diffusion.data.builders import build_inpaint_dataset
from disco.core.latent_diffusion.data.datasets import InpaintDataset
from disco.core.pixel_diffusion.precompute.base import PixelPrecomputeStrategy

@dataclass
class InpaintPrecomputeStrategy(PixelPrecomputeStrategy):
    masks_per_image_train: int = 2
    masks_per_image_val: int = 5
    img_size: int = 512
    def build_dataset(self, root_dir: Path):
        return build_inpaint_dataset(
            root_dir=root_dir,
            masks_per_image_train=self.masks_per_image_train,
            masks_per_image_val=self.masks_per_image_val,
            img_size=self.img_size
        )
    def get_encoder_input(self, batch: Any) -> torch.Tensor:
        masked_img, img, mask = batch
        return masked_img

    def get_target_img(self, batch: Any) -> torch.Tensor:
        masked_img, img, mask = batch
        return img

    def get_sample_name(
        self,
        *,
        dataset: InpaintDataset,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> str:
        return f"{split_name}_{global_idx:07d}"

    def get_metadata(
        self,
        *,
        dataset: InpaintDataset,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> dict[str, Any]:
        masked_img, img, mask = batch
        return {
            "mask": mask[batch_idx].detach().cpu(),
        }