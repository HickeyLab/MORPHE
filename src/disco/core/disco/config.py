from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from enum import StrEnum

from disco.core.autoencoder.artifact import AutoencoderArtifact
from disco.core.gcnn.artifact import GCNNArtifact
from disco.core.latent_diffusion.artifact import LatentDiffusionArtifact
from disco.core.pixel_diffusion.artifact import PixelDiffusionTrainerArtifact


@dataclass(frozen=True, slots=True)
class DiscoConfig:
    seed: Optional[int] = 0
    autoencoder: AutoencoderArtifact = None
    gcnn: GCNNArtifact = None
    latent_diffuser: LatentDiffusionArtifact = None
    pixel_diffuser: PixelDiffusionTrainerArtifact = None