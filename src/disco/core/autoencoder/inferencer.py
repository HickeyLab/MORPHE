from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from disco.core.autoencoder.artifact import AutoencoderArtifact
from disco.core.autoencoder.model import Autoencoder
from disco.core.autoencoder.utils.rasterize import rasterize_rgb_regions

@dataclass(frozen=True)
class AutoencoderRGBInferencer:
    """
    Structural wrapper around the existing functions.
    Intentionally does NOT change the underlying inference logic.
    """
    artifact: "AutoencoderArtifact"
    model: "Autoencoder"
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

    def encode_to_rgb(self, emb_matrix: torch.Tensor | np.ndarray) -> torch.Tensor:
        z_min, z_max = self.artifact.z_min, self.artifact.z_max

        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)

        with torch.no_grad():
            z = self.model.encoder(emb_matrix.to(self.device, dtype=torch.float32))

            z_min = z_min.to(self.device)
            z_max = z_max.to(self.device)

            range_vals = z_max - z_min
            range_vals[range_vals == 0] = 1e-9

            scaled_3d = (z - z_min) / range_vals
            scaled_3d = torch.clamp(scaled_3d, 0.0, 1.0)

            rgb_3d = (scaled_3d * 255).round().to(torch.uint8)

        return rgb_3d.cpu()

    def encode_to_rgb_df(
        self,
        df: pd.DataFrame,
        *,
        r_col: str = "R",
        g_col: str = "G",
        b_col: str = "B",
    ) -> pd.DataFrame:
        emb_matrix = df.loc[:, self.artifact.input_cols].to_numpy(dtype=np.float32, copy=True)
        rgb_3d = self.encode_to_rgb(emb_matrix)

        out = df.copy()
        out[r_col] = rgb_3d[:, 0].numpy()
        out[g_col] = rgb_3d[:, 1].numpy()
        out[b_col] = rgb_3d[:, 2].numpy()
        return out

    def decode_rgb_to_logits(
        self,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        z_min, z_max = self.artifact.z_min, self.artifact.z_max

        rgb = rgb.to(self.device, dtype=torch.float32)
        z_min = z_min.to(self.device)
        z_max = z_max.to(self.device)

        range_vals = z_max - z_min
        range_vals[range_vals == 0] = 1e-9

        with torch.no_grad():
            if rgb.ndim == 2:
                z_recovered = rgb * range_vals + z_min
                logits = self.model.decoder(z_recovered)
                return logits

            elif rgb.ndim == 3:
                flat_img = rgb.permute(1, 2, 0).reshape(-1, 3)
                white_mask = (flat_img == 1.0).all(dim=1)

                logits = torch.zeros(
                    flat_img.shape[0],
                    self.artifact.in_dim,
                    device=self.device,
                )

                infer_input_rgb = flat_img[~white_mask]
                if infer_input_rgb.shape[0] > 0:
                    z_recovered = infer_input_rgb * range_vals + z_min
                    logits_valid = self.model.decoder(z_recovered)
                    logits[~white_mask] = logits_valid

                h, w = rgb.shape[1], rgb.shape[2]
                return logits.reshape(h, w, self.artifact.in_dim)

            else:
                raise ValueError(f"Expected rgb to have ndim 2 or 3, got shape {rgb.shape}.")

    def decode_rgb_to_cell_map(
        self,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        rgb = rgb.to(self.device, dtype=torch.float32)
        h, w = rgb.shape[1], rgb.shape[2]

        flat_img = rgb.permute(1, 2, 0).reshape(-1, 3)
        white_mask = (flat_img == 1.0).all(dim=1)

        pred = torch.zeros(flat_img.shape[0], dtype=torch.long, device=self.device)

        logits = self.decode_rgb_to_logits(rgb)
        logits = logits.reshape(-1, self.artifact.in_dim)

        if (~white_mask).any():
            pred[~white_mask] = torch.argmax(logits[~white_mask], dim=1) + 1

        pred = pred.reshape(1, h, w)
        return pred

    def add_rgb_and_rasterize_per_region(
        self,
        result_df: pd.DataFrame,
        save_dir: str | Path,
    ) -> pd.DataFrame:
        result_df = self.encode_to_rgb_df(result_df)

        rasterize_rgb_regions(result_df, save_dir)

        return result_df