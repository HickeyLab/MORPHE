from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from ..preprocessor.config import PreProcessConfig
from ..autoencoder.artifact import AutoencoderArtifact
from ..gcnn.artifact import GCNNArtifact
from ..latent_diffusion.artifact import LatentDiffusionArtifact
from ..pixel_diffusion.artifact import PixelDiffusionArtifact


@dataclass(frozen=True, slots=True)
class MorpheArtifact:
    """
    Serializable container bundling all sub-artifacts required to build a
    full MORPHE inference pipeline.
    """

    preprocessor_config: PreProcessConfig | None = None
    autoencoder_artifact: AutoencoderArtifact | None = None
    gcnn_artifact: GCNNArtifact | None = None
    latent_diffusion_artifact: LatentDiffusionArtifact | None = None
    pixel_diffusion_artifact: PixelDiffusionArtifact | None = None

    @property
    def region_col(self) -> str:
        """Region column name from the preprocessor config."""
        if self.preprocessor_config is None:
            return "unique_region"
        return self.preprocessor_config.region_col

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
        Load a serialized ``MorpheArtifact`` from disk.

        Args:
            path: Path to a previously saved artifact file.

        Returns:
            Loaded ``MorpheArtifact`` instance.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
            TypeError: If the loaded object is not a ``MorpheArtifact``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MorpheArtifact file not found: {path}")

        obj = torch.load(path, map_location="cpu")

        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}.")

        return obj
