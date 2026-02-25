from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL

from src.disco.core.latent_diffusion.model import CondEncoder, CoordEncoder, CondEncoder3D, BBoxEncoder
from src.disco.core.latent_diffusion.strategy.base import DiffusionStrategy

@dataclass(frozen=True)
class LatentDiffusionRuntime:
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    cond_proj: CondEncoder
    coord_encoder: CoordEncoder | None
    bbox_encoder: BBoxEncoder | None
    scaling_factor: float

    
@dataclass(frozen=True)
class LatentDiffuserArtifact:
    unet_state_dict: Mapping[str, torch.Tensor]
    cond_encoder_state_dict: Mapping[str, torch.Tensor]
    coord_encoder_state_dict: Mapping[str, torch.Tensor] | None
    bbox_encoder_state_dict: Mapping[str, torch.Tensor] | None
    unet_config: Mapping[str, Any]
    cond_encoder_kwargs: Mapping[str, Any]
    coord_encoder_kwargs: Mapping[str, Any] | None
    bbox_encoder_kwargs: Mapping[str, Any] | None

    def build_components(
        self,
        *,
        strategy: DiffusionStrategy,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[UNet2DConditionModel, CondEncoder, CoordEncoder | None]:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        unet = UNet2DConditionModel.from_config(dict(self.unet_config))
        unet.load_state_dict(self.unet_state_dict, strict=True)
        
        requires_coord = bool(getattr(strategy, "requires_coord_encoder", False))
        requires_bbox = bool(getattr(strategy, "requires_bbox_encoder", False))
        coord_enc = None
        bbox_enc = None
        if requires_bbox:
            if self.bbox_encoder_state_dict is None or self.bbox_encoder_kwargs is None:
                raise ValueError(
                    "Artifact missing CoordEncoder data but strategy.requires_coord_encoder=True"
                )
            bbox_enc = BBoxEncoder(**dict(self.bbox_encoder_kwargs))
            bbox_enc.load_state_dict(self.bbox_encoder_state_dict, strict=True)
        elif requires_coord:
            if self.coord_encoder_state_dict is None or self.coord_encoder_kwargs is None:
                raise ValueError(
                    "Artifact missing CoordEncoder data but strategy.requires_coord_encoder=True"
                )
            coord_enc = CoordEncoder(**dict(self.coord_encoder_kwargs))
            coord_enc.load_state_dict(self.coord_encoder_state_dict, strict=True)
        
        cond_enc = CondEncoder3D(**dict(self.cond_encoder_kwargs)) if strategy.three_dimensional_cond_encoder else CondEncoder(**dict(self.cond_encoder_kwargs))
        cond_enc.load_state_dict(self.cond_encoder_state_dict, strict=True)

        if dtype is not None:
            unet = unet.to(device=device, dtype=dtype)
            cond_enc = cond_enc.to(device=device, dtype=dtype)
            if coord_enc is not None:
                coord_enc = coord_enc.to(device=device, dtype=dtype)
            if bbox_enc is not None:
                coord_enc = bbox_enc.to(device=device, dtype=dtype)
        else:
            unet = unet.to(device)
            cond_enc = cond_enc.to(device)
            if coord_enc is not None:
                coord_enc = coord_enc.to(device)
            if bbox_enc is not None:
                coord_enc = bbox_enc.to(device)

        unet.eval()
        cond_enc.eval()
        if coord_enc is not None:
            coord_enc.eval()
        if bbox_enc is not None:
            bbox_enc.eval()

        return unet, cond_enc, coord_enc, bbox_enc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "unet_state_dict": {k: v.detach().cpu() for k, v in self.unet_state_dict.items()},
            "cond_encoder_state_dict": {k: v.detach().cpu() for k, v in self.cond_encoder_state_dict.items()},
            "coord_encoder_state_dict": (
                None
                if self.coord_encoder_state_dict is None
                else {k: v.detach().cpu() for k, v in self.coord_encoder_state_dict.items()}
            ),
            "bbox_encoder_state_dict": (
                None
                if self.bbox_encoder_state_dict is None
                else {k: v.detach().cpu() for k, v in self.bbox_encoder_state_dict.items()}
            ),
            "unet_config": dict(self.unet_config),
            "cond_encoder_kwargs": dict(self.cond_encoder_kwargs),
            "coord_encoder_kwargs": None if self.coord_encoder_kwargs is None else dict(self.coord_encoder_kwargs),
            "bbox_encoder_kwargs": None if self.bbox_encoder_kwargs is None else dict(self.bbox_encoder_kwargs),
        }

        torch.save(payload, str(path))

    @staticmethod
    def load(path: str | Path) -> "LatentDiffuserArtifact":
        path = Path(path)

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")

        required = (
            "unet_state_dict",
            "cond_encoder_state_dict",
            "coord_encoder_state_dict",
            "bbox_encoder_state_dict",
            "unet_config",
            "cond_encoder_kwargs",
            "coord_encoder_kwargs",
            "bbox_encoder_kwargs",
        )
        for k in required:
            if k not in payload:
                raise ValueError(f"Invalid LatentDiffuserArtifact file: missing key '{k}'")

        coord_sd = payload["coord_encoder_state_dict"]
        coord_kwargs = payload["coord_encoder_kwargs"]
        bbox_sd = payload["bbox_encoder_state_dict"]
        bbox_kwargs = payload["bbox_encoder_kwargs"]

        return LatentDiffuserArtifact(
            unet_state_dict=payload["unet_state_dict"],
            cond_encoder_state_dict=payload["cond_encoder_state_dict"],
            coord_encoder_state_dict=None if coord_sd is None else coord_sd,
            bbox_encoder_state_dict=None if bbox_sd is None else bbox_sd,
            unet_config=dict(payload["unet_config"]),
            cond_encoder_kwargs=dict(payload["cond_encoder_kwargs"]),
            coord_encoder_kwargs=None if coord_kwargs is None else dict(coord_kwargs),
            bbox_encoder_kwargs=None if bbox_kwargs is None else dict(bbox_kwargs),
        )
    
    def build_inference_runtime(
        self,
        *,
        strategy: DiffusionStrategy,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LatentDiffusionRuntime:
        """
        Build everything needed for inference:
          - VAE from diffusers pretrained
          - Scheduler from diffusers pretrained
          - UNet/CondEncoder/CoordEncoder from artifact weights/config
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        unet, cond_enc, coord_enc, bbox_enc = self.build_components(
            strategy=strategy, device=device, dtype=dtype,
        )

        vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae").to(device)
        if dtype is not None:
            vae = vae.to(device=device, dtype=dtype)
        vae.eval()

        scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

        noise_scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")

        #ONLY SLICE3D HAS THIS
        if not strategy.requires_bbox_encoder and not strategy.requires_bbox_encoder:
            if hasattr(noise_scheduler, "config"):
                noise_scheduler.config.prediction_type = "sample"

        return LatentDiffusionRuntime(
            vae=vae,
            unet=unet,
            cond_proj=cond_enc,
            coord_encoder=coord_enc,
            bbox_encoder=bbox_enc,
            noise_scheduler=noise_scheduler,
            scaling_factor=scaling_factor,
        )
