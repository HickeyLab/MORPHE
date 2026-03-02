from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .inferencer import DiscoInferencer


@dataclass(frozen=True, slots=True)
class DiscoArtifact:
    """
    Serializable container bundling all sub-artifacts required
    for inference.
    """

    autoencoder: Any | None = None
    gcnn: Any | None = None
    latent_diffuser: Any | None = None
    pixel_diffuser: Any | None = None

    def save(self, path: str | Path) -> None:
        """Serialize artifact to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @staticmethod
    def load(path: str | Path) -> "DiscoArtifact":
        """Load artifact from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DiscoArtifact file not found: {path}")
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, DiscoArtifact):
            raise TypeError("Loaded object is not a DiscoArtifact.")
        return obj

    def build_inferencer(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        ld_kwargs: dict[str, Any] | None = None,
        pd_kwargs: dict[str, Any] | None = None,
        ae_kwargs: dict[str, Any] | None = None,
        gcnn_kwargs: dict[str, Any] | None = None,
    ) -> DiscoInferencer:
        """
        Construct a DiscoInferencer from this artifact.
        """
        device = torch.device(device)
        dtype = dtype or torch.float32

        ld = (
            self.latent_diffuser.build_inferencer(device=device, dtype=dtype, **(ld_kwargs or {}))
            if self.latent_diffuser is not None
            else None
        )

        pd = (
            self.pixel_diffuser.build_inferencer(device=device, dtype=dtype, **(pd_kwargs or {}))
            if self.pixel_diffuser is not None
            else None
        )

        ae = (
            self.autoencoder.build_inferencer(device=device, dtype=dtype, **(ae_kwargs or {}))
            if self.autoencoder is not None
            else None
        )

        gcnn = (
            self.gcnn.build_inferencer(device=device, dtype=dtype, **(gcnn_kwargs or {}))
            if self.gcnn is not None
            else None
        )

        return DiscoInferencer(
            ae=ae,
            gcnn=gcnn,
            ld=ld,
            pd=pd,
            device=device,
            dtype=dtype,
        )