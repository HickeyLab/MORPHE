from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import torch
from disco.core.latent_diffusion.infer.run_config import LatentBaseRunConfig
from disco.core.latent_diffusion.train.base import LatentTrainStrategy
from src.disco.core.latent_diffusion.artifact import LatentDiffusionArtifact, LatentDiffusionRuntime


RunConfigT = TypeVar("RunConfigT", bound=LatentBaseRunConfig)
TStrategy = TypeVar("TStrategy", bound=LatentTrainStrategy)
class BaseLatentInferencer(ABC, Generic[RunConfigT, TStrategy]):
    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        train_strategy: TStrategy,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        rt: LatentDiffusionRuntime = artifact.build_inference_runtime(
            train_strategy=train_strategy,
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )
        self.train_strategy = train_strategy

        self.vae = rt.vae
        self.unet = rt.unet
        self.noise_scheduler = rt.noise_scheduler
        self.coord_encoder = rt.coord_encoder
        self.bbox_encoder = rt.bbox_encoder
        self.cond_proj = rt.cond_proj
        self.scaling_factor = rt.scaling_factor
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
    @classmethod
    def from_artifact(
        cls,
        artifact: LatentDiffusionArtifact,
        *,
        train_strategy: TStrategy,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BaseLatentInferencer[RunConfigT, TStrategy]:
        if device is not None:
            device = torch.device(device)

        return cls(
            artifact=artifact,
            train_strategy=train_strategy,
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )


    @torch.no_grad()
    def __call__(self, *args: Any, **kwargs: Any) -> list[torch.Tensor]:
        return self.run(*args, **kwargs)

    @abstractmethod
    @torch.no_grad()
    def run(self, cfg: RunConfigT) -> list[torch.Tensor]:
        """Implemented by Gapfill / Inpaint / Slice3D / etc."""
        raise NotImplementedError
