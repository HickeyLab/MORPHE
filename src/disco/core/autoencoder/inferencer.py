from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from src.disco.core.autoencoder.artifact import AutoencoderArtifact
from src.disco.core.autoencoder.model import Autoencoder


@dataclass(frozen=True)
class AutoencoderRGBInferencer:
    """
    Structural wrapper around the existing functions.
    Intentionally does NOT change the underlying inference logic.
    """
    artifact: "AutoencoderArtifact"
    model: Autoencoder
    device: torch.device

    @classmethod
    def from_artifact(
        cls,
        artifact: "AutoencoderArtifact",
        *,
        device: torch.device | str | None = None,
    ) -> "AutoencoderRGBInferencer":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model = artifact.build_model(device=device)
        
        return cls(
            artifact=artifact,
            model=model,
            device=device,
        )

    def encode_to_rgb(self, emb_matrix: torch.Tensor) -> torch.Tensor:
        if not self.device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)

        z_min, z_max = self.artifact.z_min, self.artifact.z_max
        with torch.no_grad():
            z = self.model.encoder(emb_matrix.to(device))

            # ensure same device
            z_min = z_min.to(device)
            z_max = z_max.to(device)

            range_vals = (z_max - z_min)
            range_vals[range_vals == 0] = 1e-9

            scaled_3d = (z - z_min) / range_vals
            scaled_3d = torch.clamp(scaled_3d, 0.0, 1.0)

            rgb_3d = (scaled_3d * 255).round().to(torch.uint8)

        return rgb_3d.cpu()

    def encode_to_rgb_df(
        self,
        df: pd.DataFrame,
        emb_matrix: np.ndarray,
        *,
        r_col: str = "R",
        g_col: str = "G",
        b_col: str = "B",
    ) -> pd.DataFrame:
        rgb_3d = self.encode_to_rgb(emb_matrix)
        
        out = df.copy()
        out[r_col] = rgb_3d[:, 0].numpy()
        out[g_col] = rgb_3d[:, 1].numpy()
        out[b_col] = rgb_3d[:, 2].numpy()
        return out