from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import torch

from disco.core.autoencoder.model import Autoencoder


@dataclass(frozen=True)
class AutoencoderArtifact:
    state_dict: Mapping[str, torch.Tensor]
    in_dim: int
    bottleneck_dim: int
    hidden_dim: int
    z_min: torch.Tensor
    z_max: torch.Tensor

    def build_model(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> Autoencoder:
        model = Autoencoder(
            in_dim=self.in_dim,
            bottleneck_dim=self.bottleneck_dim,
            hidden_dim=self.hidden_dim
        )
        model.load_state_dict(self.state_dict, strict=True)
        
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        model = model.to(device)
        model.eval()
        return model

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict,
                "in_dim": self.in_dim,
                "bottleneck_dim": self.bottleneck_dim,
                "hidden_dim": self.hidden_dim,
                "z_min": self.z_min,
                "z_max": self.z_max
            },
            str(path),
        )


    @staticmethod
    def load(path: str | Path) -> "AutoencoderArtifact":
        payload = torch.load(path, map_location="cpu")

        return AutoencoderArtifact(
            state_dict=payload["state_dict"],
            in_dim=payload["in_dim"],
            bottleneck_dim=payload["bottleneck_dim"],
            hidden_dim=payload["hidden_dim"],
            z_min=payload["z_min"],
            z_max=payload["z_max"]
        )