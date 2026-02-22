import numpy as np
import pandas as pd
import torch
from disco.core.autoencoder.artifact import AutoencoderArtifact
from disco.core.autoencoder.model import Autoencoder


def encode_to_rgb(
    model: Autoencoder,
    emb_matrix: np.ndarray,
    *,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    with torch.no_grad():
        z = model.encoder(emb_matrix.to(device))

    # scale to [0,1]
    range_vals = (z_max - z_min)
    range_vals[range_vals == 0] = 1e-9
    scaled_3d = (z - z_min) / range_vals

    # convert to RGB
    rgb_3d = (scaled_3d * 255).round().clamp(0, 255).to(torch.uint8)
    return rgb_3d.cpu()


def add_rgb_to_df(
    df: pd.DataFrame,
    rgb_3d: torch.Tensor,
    *,
    r_col: str = "R",
    g_col: str = "G",
    b_col: str = "B",
) -> pd.DataFrame:
    out = df.copy()
    out[r_col] = rgb_3d[:, 0].numpy()
    out[g_col] = rgb_3d[:, 1].numpy()
    out[b_col] = rgb_3d[:, 2].numpy()
    return out

class AutoencoderRGBInferencer:
    """
    Structural wrapper around the existing functions.
    Intentionally does NOT change the underlying inference logic.
    """

    def __init__(
        self,
        artifact: AutoencoderArtifact,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.artifact = artifact
        self.device = device

    def encode_to_rgb(self, emb_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)

        model = self.artifact.build_model(device=device)
        return encode_to_rgb(
            model,
            emb_matrix,
            z_min=self.artifact.z_min,
            z_max=self.artifact.z_max,
            device=device,
        )

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
        return add_rgb_to_df(df, rgb_3d, r_col=r_col, g_col=g_col, b_col=b_col)
