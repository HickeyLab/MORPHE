# disco/core/disco/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class DiscoConfig:
    seed: Optional[int] = 0
    autoencoder: Any = None
    gcnn: Any = None
    latent_diffuser: Any = None
    pixel_diffuser: Any = None