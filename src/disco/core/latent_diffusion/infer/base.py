from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from disco.core.latent_diffusion.artifact import LatentDiffuserArtifact, LatentDiffusionRuntime
from disco.core.latent_diffusion.strategy.base import DiffusionStrategy

@dataclass(frozen=True)
class InferenceResult:
    image: torch.Tensor
    latents: Optional[torch.Tensor] = None
    extras: dict[str, Any] = None
    
class LatentDiffusionInferencer(ABC):
    def __init__(
        self,
        *,
        artifact: LatentDiffuserArtifact,
        strategy: DiffusionStrategy,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.strategy = strategy

        rt: LatentDiffusionRuntime = artifact.build_inference_runtime(
            strategy=strategy,
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )

        self.vae = rt.vae
        self.unet = rt.unet
        self.noise_scheduler = rt.noise_scheduler
        self.coord_encoder = rt.coord_encoder
        self.cond_proj = rt.cond_proj
        self.scaling_factor = rt.scaling_factor
        self.device = device

    @torch.no_grad()
    def __call__(self, *args: Any, **kwargs: Any) -> InferenceResult:
        return self.run(*args, **kwargs)

    @abstractmethod
    @torch.no_grad()
    def run(self, *args: Any, **kwargs: Any) -> InferenceResult:
        """Implemented by Gapfill / Inpaint / Slice3D / etc."""
        raise NotImplementedError
