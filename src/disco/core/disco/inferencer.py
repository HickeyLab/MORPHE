from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class DiscoInferencer:
    """
    High-level inference orchestrator.

    Delegates generation to latent diffusion and decoding/refinement
    to pixel diffusion or autoencoder.
    """

    ae: Any | None
    gcnn: Any | None
    ld: Any | None
    pd: Any | None
    device: torch.device
    dtype: torch.dtype

    # -------------------------
    # Internal helpers
    # -------------------------

    def _require_ld(self) -> Any:
        if self.ld is None:
            raise RuntimeError(
                "Latent diffusion inferencer is not available. "
                "Ensure latent_diffuser was trained or provided."
            )
        if not hasattr(self.ld, "generate"):
            raise AttributeError("Latent diffusion inferencer lacks `generate` method.")
        return self.ld

    def _decode(self, latents: Any, **kwargs: Any) -> Any:
        if self.pd is not None:
            if not hasattr(self.pd, "decode"):
                raise AttributeError("Pixel diffusion inferencer lacks `decode` method.")
            return self.pd.decode(latents, **kwargs)

        if self.ae is not None:
            if not hasattr(self.ae, "decode"):
                raise AttributeError("Autoencoder inferencer lacks `decode` method.")
            return self.ae.decode(latents, **kwargs)

        raise RuntimeError(
            "No decoder available. Provide pixel_diffuser or autoencoder artifact."
        )

    def _run_task(self, *, strategy: Any, **kwargs: Any) -> Any:
        ld = self._require_ld()
        latents = ld.generate(strategy=strategy, **kwargs)
        return self._decode(latents)

    # -------------------------
    # Public task methods
    # -------------------------

    def inpaint(self, *, strategy: Any, **kwargs: Any) -> Any:
        return self._run_task(strategy=strategy, **kwargs)

    def outpaint(self, *, strategy: Any, **kwargs: Any) -> Any:
        return self._run_task(strategy=strategy, **kwargs)

    def impute_2d(self, *, strategy: Any, **kwargs: Any) -> Any:
        return self._run_task(strategy=strategy, **kwargs)

    def impute_3d(self, *, strategy: Any, **kwargs: Any) -> Any:
        return self._run_task(strategy=strategy, **kwargs)