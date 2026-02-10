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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cpu_state = {k: v.detach().cpu() for k, v in self.state_dict.items()}
        payload = {
            "state_dict": cpu_state,
            "in_dim": self.in_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "hidden_dim": self.hidden_dim,
            "z_min": self.z_min.detach().cpu(),
            "z_max": self.z_max.detach().cpu(),
        }
        torch.save(payload, str(path))


    @staticmethod
    def load(path: str | Path) -> "AutoencoderArtifact":
        path = Path(path)
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")

        for k in ("state_dict", "in_dim", "bottleneck_dim", "hidden_dim", "z_min", "z_max"):
            if k not in payload:
                raise ValueError(f"Invalid artifact file: missing key '{k}'")

        return AutoencoderArtifact(
            state_dict=payload["state_dict"],
            in_dim=int(payload["in_dim"]),
            bottleneck_dim=int(payload["bottleneck_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            z_min=payload["z_min"],
            z_max=payload["z_max"],
        )