from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel # type: ignore

from .run_config import LatentBaseRunConfig
from ..model import BBoxEncoder, CondEncoder, CondEncoder3D, CoordEncoder
from ....utils import resolve_device, resolve_dtype
from ..artifact import LatentDiffusionArtifact

def _get_scaling_factor(vae: AutoencoderKL) -> float:
    return float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

RunConfigT = TypeVar("RunConfigT", bound=LatentBaseRunConfig)
class BaseLatentInferencer(ABC, Generic[RunConfigT]):
    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        cond_encoder: CondEncoder | CondEncoder3D,
        coord_encoder: CoordEncoder | None,
        bbox_encoder: BBoxEncoder | None,
        noise_scheduler: DDPMScheduler,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        self.artifact = artifact
        self.unet = unet
        self.vae = vae
        self.cond_encoder = cond_encoder
        self.coord_encoder = coord_encoder
        self.bbox_encoder = bbox_encoder
        self.noise_scheduler = noise_scheduler
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(device=self.device, dtype=dtype)

        self.unet.eval()
        self.vae.eval()
        self.cond_encoder.eval()
        if self.coord_encoder is not None:
            self.coord_encoder.eval()
        if self.bbox_encoder is not None:
            self.bbox_encoder.eval()

        self.scaling_factor = _get_scaling_factor(self.vae)
        
    @classmethod
    @abstractmethod
    def from_artifact(
        cls,
        artifact: LatentDiffusionArtifact,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        raise NotImplementedError
    
    @abstractmethod
    @torch.no_grad()
    def _run_one_from_config(self, cfg: RunConfigT) -> torch.Tensor:
        """Run inference from a validated task-specific config."""
        raise NotImplementedError