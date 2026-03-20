from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path
from typing import Any

import torch

from disco.core.latent_diffusion.data.builders import build_outpaint_dataset
from disco.core.latent_diffusion.data.datasets import OutpaintDataset
from disco.core.pixel_diffusion.precompute.base import PixelPrecomputeStrategy

@dataclass
class OutpaintPrecomputeStrategy(PixelPrecomputeStrategy):
    masks_per_image_train: int = 5
    masks_per_image_val: int = 5
    img_size: int = 512
    
    def build_dataset(self, root_dir: Path):
        return build_outpaint_dataset(
            root_dir=root_dir,
            masks_per_image_train=self.masks_per_image_train,
            masks_per_image_val=self.masks_per_image_val,
            img_size=self.img_size
        )
    def get_encoder_input(self, batch: Any) -> torch.Tensor:
        masked_img, img, bbox = batch
        return masked_img

    def get_target_img(self, batch: Any) -> torch.Tensor:
        masked_img, img, bbox = batch
        return img

    def get_sample_name(
        self,
        *,
        dataset: OutpaintDataset,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> str:
        if hasattr(dataset, "img_files") and hasattr(dataset, "masks_per_image"):
            img_idx = global_idx // dataset.masks_per_image
            k = global_idx % dataset.masks_per_image
            base = Path(dataset.img_files[img_idx]).stem
            return f"{base}_k{k:03d}"

        return f"{split_name}_{global_idx:07d}"

    def get_metadata(
        self,
        *,
        dataset: OutpaintDataset,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> dict[str, Any]:
        masked_img, img, bbox = batch
        return {
            "bbox": bbox[batch_idx].detach().cpu(),
        }