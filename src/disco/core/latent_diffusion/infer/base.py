from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

import torch
from diffusers import AutoencoderKL # type: ignore

from disco.core.latent_diffusion.infer.run_config import LatentBaseRunConfig
from disco.utils import resolve_device, resolve_dtype
from disco.core.latent_diffusion.artifact import LatentDiffusionArtifact

def _get_scaling_factor(vae: AutoencoderKL) -> float:
    return float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

RunConfigT = TypeVar("RunConfigT", bound=LatentBaseRunConfig)
class BaseLatentInferencer(ABC, Generic[RunConfigT]):
    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.artifact = artifact
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        
        (
            self.unet,
            self.vae,
            self.cond_encoder,
            self.coord_encoder,
            self.bbox_encoder,
            self.noise_scheduler,
        ) = self.artifact.architecture.build_components(
            device=self.device,
            dtype=self.dtype,
        )
        
        self.unet.eval()
        self.vae.eval()
        self.cond_encoder.eval()
        if self.coord_encoder is not None:
            self.coord_encoder.eval()
        if self.bbox_encoder is not None:
            self.bbox_encoder.eval()
            
        self.scaling_factor = _get_scaling_factor(self.vae)
        
    @classmethod
    def from_artifact(
        cls,
        artifact: LatentDiffusionArtifact,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        return cls(
            artifact=artifact,
            device=device,
            dtype=dtype,
        )

    @abstractmethod
    @torch.no_grad()
    def _run_one_from_config(self, cfg: RunConfigT) -> torch.Tensor:
        """Run inference from a validated task-specific config."""
        raise NotImplementedError