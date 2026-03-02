from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch


@dataclass(frozen=True)
class DiscoResult:
    """
    Standard output for DISCO high-level inference.

    rgb_embedding:
      - RGB embedding image (typically 3xHxW) in whatever range your pipeline uses
        (often [-1, 1] for diffusion, or [0,1]/[0,255] for visualization).

    logits:
      - Cell-type logits/probabilities decoded from rgb_embedding (shape depends on your convention).
    """
    rgb_embedding: torch.Tensor
    logits: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    extras: Optional[Mapping[str, Any]] = None