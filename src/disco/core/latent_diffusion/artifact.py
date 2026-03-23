from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from disco.config import InferenceMode
from disco.core.latent_diffusion.architecture import LatentArchitectureSpec
from disco.core.latent_diffusion.infer.base import BaseLatentInferencer
from disco.core.latent_diffusion.infer.registry import _LATENT_INFERENCER_REGISTRY


@dataclass(frozen=True, slots=True)
class LatentDiffusionArtifact:
    """
    Serializable artifact for reconstructing a trained latent diffusion model.

    The artifact stores trained weights plus the architecture specification
    required to rebuild the exact model structure for inference.
    """

    unet_state_dict: dict[str, torch.Tensor]
    cond_encoder_state_dict: dict[str, torch.Tensor]
    coord_encoder_state_dict: dict[str, torch.Tensor] | None
    bbox_encoder_state_dict: dict[str, torch.Tensor] | None
    architecture: LatentArchitectureSpec
    inference_mode: InferenceMode
    
    img_size: int | None = None  # Optional image size metadata for inference preprocessing (only inpainting/outpainting)

    _REQUIRED_KEYS = (
        "unet_state_dict",
        "cond_encoder_state_dict",
        "coord_encoder_state_dict",
        "bbox_encoder_state_dict",
        "architecture",
        "inference_mode",
        "img_size",
    )

    @staticmethod
    def _state_dict_to_cpu(
        state_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor] | None:
        """
        Return a CPU, detached copy of a state dict for device-agnostic serialization.
        """
        if state_dict is None:
            return None

        return {
            key: value.detach().cpu()
            for key, value in state_dict.items()
        }

    def save(self, path: str | Path) -> None:
        """
        Save the artifact payload to disk.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "unet_state_dict": self._state_dict_to_cpu(self.unet_state_dict),
            "cond_encoder_state_dict": self._state_dict_to_cpu(self.cond_encoder_state_dict),
            "coord_encoder_state_dict": self._state_dict_to_cpu(self.coord_encoder_state_dict),
            "bbox_encoder_state_dict": self._state_dict_to_cpu(self.bbox_encoder_state_dict),
            "architecture": self.architecture.to_dict(),
            "inference_mode": self.inference_mode.value,
            "img_size": self.img_size,
        }

        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "LatentDiffusionArtifact":
        """
        Load a serialized artifact from disk and validate its required fields.
        """
        path = Path(path)

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")

        if not isinstance(payload, dict):
            raise ValueError("Invalid LatentDiffusionArtifact file: payload must be a dictionary.")

        for key in cls._REQUIRED_KEYS:
            if key not in payload:
                raise ValueError(
                    f"Invalid LatentDiffusionArtifact file: missing key '{key}'."
                )

        architecture_payload = payload["architecture"]
        if not isinstance(architecture_payload, dict):
            raise ValueError(
                "Invalid LatentDiffusionArtifact file: 'architecture' must be a dictionary."
            )

        return cls(
            unet_state_dict=dict(payload["unet_state_dict"]),
            cond_encoder_state_dict=dict(payload["cond_encoder_state_dict"]),
            coord_encoder_state_dict=(
                None
                if payload["coord_encoder_state_dict"] is None
                else dict(payload["coord_encoder_state_dict"])
            ),
            bbox_encoder_state_dict=(
                None
                if payload["bbox_encoder_state_dict"] is None
                else dict(payload["bbox_encoder_state_dict"])
            ),
            architecture=LatentArchitectureSpec.from_dict(architecture_payload),
            inference_mode=InferenceMode(payload["inference_mode"]),
            img_size=payload["img_size"],
        )
        
    def build_inferencer(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BaseLatentInferencer:
        inferencer_cls = _LATENT_INFERENCER_REGISTRY[self.inference_mode]
        return inferencer_cls.from_artifact(
            artifact=self,
            device=device,
            dtype=dtype,
        )