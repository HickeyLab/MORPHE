from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from typing import Any

import torch

from disco.core.latent_diffusion.data.builders import build_three_dim_dataset
from disco.core.latent_diffusion.data.datasets import Slice3DDataset
from disco.core.pixel_diffusion.precompute.base import PixelPrecomputeStrategy

@dataclass
class ThreeDimImputationPrecomputeStrategy(PixelPrecomputeStrategy):
    def build_dataset(self, root_dir: Path):
        return build_three_dim_dataset(root_dir=root_dir)
        
    def get_encoder_input(self, batch: Any) -> torch.Tensor:
        img_prev, img_next, img_gt, w_prev, w_next = batch
        return img_prev

    def get_target_img(self, batch: Any) -> torch.Tensor:
        img_prev, img_next, img_gt, w_prev, w_next = batch
        return img_gt

    def get_sample_name(
        self,
        *,
        dataset: Slice3DDataset,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> str:
        return f"{split_name}_{global_idx:07d}"

    def get_metadata(
        self,
        *,
        dataset: Slice3DDataset,
        batch: Any,
        batch_idx: int,
        global_idx: int,
        split_name: str,
    ) -> dict[str, Any]:
        img_prev, img_next, img_gt, w_prev, w_next = batch
        return {
            "w_prev": float(w_prev[batch_idx].item()),
            "w_next": float(w_next[batch_idx].item()),
        }