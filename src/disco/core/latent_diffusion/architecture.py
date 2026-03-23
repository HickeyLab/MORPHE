from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL  # type: ignore

from disco.core.latent_diffusion.model import BBoxEncoder, CondEncoder, CondEncoder3D, CoordEncoder


@dataclass(frozen=True, slots=True)
class LatentArchitectureSpec:
    """
    Self-contained specification for reconstructing the latent diffusion
    architecture.

    This spec is intended to be stored in the artifact and reused by both
    training and inference so that architecture construction logic lives in
    exactly one place.
    """

    # Core UNet architecture config.
    unet_pretrained_path: str = "runwayml/stable-diffusion-v1-5"
    ae_pretrained_path: str = "runwayml/stable-diffusion-v1-5"
    scheduler_pretrained_path: str = "runwayml/stable-diffusion-v1-5"

    # Encoder type selectors. ``None`` means that encoder is not used.
    cond_encoder_type: str = "cond"
    coord_encoder_type: str | None = None
    bbox_encoder_type: str | None = None

    # Constructor kwargs for each encoder.
    cond_encoder_kwargs: Mapping[str, Any] | None = None
    coord_encoder_kwargs: Mapping[str, Any] | None = None
    bbox_encoder_kwargs: Mapping[str, Any] | None = None

    def build_components(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        UNet2DConditionModel,
        AutoencoderKL,
        CondEncoder | CondEncoder3D,
        CoordEncoder | None,
        BBoxEncoder | None,
        DDPMScheduler,
    ]:
        """
        Build and return all latent diffusion components in a fixed order.

        Returns:
            A tuple of:
            - UNet
            - VAE
            - VAE scaling factor
            - conditioning encoder
            - coordinate encoder, if used
            - bounding-box encoder, if used
            - noise scheduler
        """
        # UNet
        unet_obj = UNet2DConditionModel.from_pretrained(
            self.unet_pretrained_path,
            subfolder="unet",
        )
        unet = unet_obj[0] if isinstance(unet_obj, tuple) else unet_obj
        unet = unet.to(device=device, dtype=dtype)  # type: ignore[assignment]

        # VAE
        vae_obj = AutoencoderKL.from_pretrained(
            self.ae_pretrained_path,
            subfolder="vae",
        )
        vae = vae_obj[0] if isinstance(vae_obj, tuple) else vae_obj
        vae = vae.to(device=device, dtype=dtype)  # type: ignore[assignment]
        

        # Encoders
        cond_encoder = self._build_cond_encoder(device=device, dtype=dtype)
        coord_encoder = self._build_coord_encoder(device=device, dtype=dtype)
        bbox_encoder = self._build_bbox_encoder(device=device, dtype=dtype)

        # Noise scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.scheduler_pretrained_path, 
            subfolder="scheduler"
        ) 
        config = noise_scheduler.config 
        if isinstance(config, dict): 
            config["prediction_type"] = "sample" 
        else: config.prediction_type = "sample"

        return (
            unet,
            vae,
            cond_encoder,
            coord_encoder,
            bbox_encoder,
            noise_scheduler,
        )

    def _build_cond_encoder(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CondEncoder | CondEncoder3D:
        """
        Build the required conditioning encoder.

        Raises:
            ValueError: If ``cond_encoder_type`` is unknown.
        """
        if self.cond_encoder_type == "cond":
            model = CondEncoder(**(self.cond_encoder_kwargs or {}))

        elif self.cond_encoder_type == "cond3d":
            model = CondEncoder3D(**(self.cond_encoder_kwargs or {}))

        else:
            raise ValueError(f"Unknown cond_encoder_type: {self.cond_encoder_type}")

        model.to(device=device, dtype=dtype)
        return model

    def _build_coord_encoder(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CoordEncoder | None:
        """
        Build the coordinate encoder when configured.

        Returns:
            The coordinate encoder, or ``None`` if this architecture does not
            use one.

        Raises:
            ValueError: If ``coord_encoder_type`` is unknown.
        """
        if self.coord_encoder_type is None:
            return None

        if self.coord_encoder_type == "coord":
            model = CoordEncoder(**(self.coord_encoder_kwargs or {}))
        else:
            raise ValueError(f"Unknown coord_encoder_type: {self.coord_encoder_type}")
        
        model = model.to(device=device, dtype=dtype)
        
        return model

    def _build_bbox_encoder(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BBoxEncoder | None:
        """
        Build the bounding-box encoder when configured.

        Returns:
            The bounding-box encoder, or ``None`` if this architecture does not
            use one.

        Raises:
            ValueError: If ``bbox_encoder_type`` is unknown.
        """
        if self.bbox_encoder_type is None:
            return None

        if self.bbox_encoder_type == "bbox":
            model = BBoxEncoder(**(self.bbox_encoder_kwargs or {}))
        else:
            raise ValueError(f"Unknown bbox_encoder_type: {self.bbox_encoder_type}")

        model = model.to(device=device, dtype=dtype)

        return model
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this architecture spec into a plain dictionary.

        The returned payload is JSON-like and safe to store inside a model
        artifact alongside state dictionaries.
        """
        return {
            "unet_pretrained_path": self.unet_pretrained_path,
            "ae_pretrained_path": self.ae_pretrained_path,
            "scheduler_pretrained_path": self.scheduler_pretrained_path,
            "cond_encoder_type": self.cond_encoder_type,
            "coord_encoder_type": self.coord_encoder_type,
            "bbox_encoder_type": self.bbox_encoder_type,
            "cond_encoder_kwargs": dict(self.cond_encoder_kwargs or {}),
            "coord_encoder_kwargs": dict(self.coord_encoder_kwargs or {}),
            "bbox_encoder_kwargs": dict(self.bbox_encoder_kwargs or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentArchitectureSpec":
        """
        Reconstruct an architecture spec from a serialized dictionary payload.

        Raises:
            ValueError: If the payload is missing required keys.
            TypeError: If the payload is not mapping-like.
        """
        required_keys = (
            "unet_pretrained_path",
            "ae_pretrained_path",
            "scheduler_pretrained_path",
            "cond_encoder_type",
            "coord_encoder_type",
            "bbox_encoder_type",
            "cond_encoder_kwargs",
            "coord_encoder_kwargs",
            "bbox_encoder_kwargs",
        )

        for key in required_keys:
            if key not in payload:
                raise ValueError(
                    f"Invalid LatentArchitectureSpec payload: missing key '{key}'"
                )

        return cls(
            unet_pretrained_path=str(payload["unet_pretrained_path"]),
            ae_pretrained_path=str(payload["ae_pretrained_path"]),
            scheduler_pretrained_path=str(payload["scheduler_pretrained_path"]),
            cond_encoder_type=str(payload["cond_encoder_type"]),
            coord_encoder_type=(
                None
                if payload["coord_encoder_type"] is None
                else str(payload["coord_encoder_type"])
            ),
            bbox_encoder_type=(
                None
                if payload["bbox_encoder_type"] is None
                else str(payload["bbox_encoder_type"])
            ),
            cond_encoder_kwargs=dict(payload["cond_encoder_kwargs"]),
            coord_encoder_kwargs=dict(payload["coord_encoder_kwargs"]),
            bbox_encoder_kwargs=dict(payload["bbox_encoder_kwargs"]),
        )