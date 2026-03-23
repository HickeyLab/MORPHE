from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from disco.config import PreProcessConfig
from disco.core.autoencoder.artifact import AutoencoderArtifact
from disco.core.gcnn.artifact import GCNNArtifact
from disco.core.latent_diffusion.artifact import LatentDiffusionArtifact
from disco.core.pixel_diffusion.artifact import PixelDiffusionArtifact
from disco.utils import resolve_device, resolve_dtype

from .inferencer import DiscoInferencer


@dataclass(frozen=True, slots=True)
class DiscoArtifact:
    """
    Serializable container bundling all sub-artifacts required to build a
    full DISCO inference pipeline.
    """

    preprocessor_config: PreProcessConfig | None = None
    autoencoder_artifact: AutoencoderArtifact | None = None
    gcnn_artifact: GCNNArtifact | None = None
    latent_diffusion_artifact: LatentDiffusionArtifact | None = None
    pixel_diffusion_artifact: PixelDiffusionArtifact | None = None

    def save(self, path: str | Path) -> None:
        """
        Serialize this artifact to disk.

        Args:
            path: Destination path for the serialized artifact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load a serialized ``DiscoArtifact`` from disk.

        Args:
            path: Path to a previously saved artifact file.

        Returns:
            Loaded ``DiscoArtifact`` instance.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
            TypeError: If the loaded object is not a ``DiscoArtifact``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DiscoArtifact file not found: {path}")

        obj = torch.load(path, map_location="cpu")

        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}.")

        return obj

    def build_inferencer(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> DiscoInferencer:
        """
        Construct a ``DiscoInferencer`` from this artifact.

        Args:
            device: Target device for runtime inference components.
            dtype: Target floating-point dtype for inference components. When
                omitted, a device-appropriate dtype is resolved automatically.

        Returns:
            Fully constructed ``DiscoInferencer``.
        """
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(device=resolved_device, dtype=dtype)

        return DiscoInferencer.from_artifact(
            artifact=self,
            device=resolved_device,
            dtype=resolved_dtype,
        )