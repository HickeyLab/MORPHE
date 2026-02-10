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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def encode_to_rgb_from_artifact(
    artifact: AutoencoderArtifact, 
    emb_matrix: np.ndarray, 
    *, 
    device: torch.device | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    model = artifact.build_model(device=device)
    return encode_to_rgb(
        model,
        emb_matrix,
        z_min=artifact.z_min,
        z_max=artifact.z_max,
        device=device,
    )


def add_rgb_to_df(
    df: pd.DataFrame, 
    rgb_3d: torch.Tensor,
    *,
    r_col: str = "R",
    g_col: str = "G",
    b_col: str = "B"
) -> pd.DataFrame:
    out = df.copy()
    out[r_col] = rgb_3d[:, 0].numpy()
    out[g_col] = rgb_3d[:, 1].numpy()
    out[b_col] = rgb_3d[:, 2].numpy()
    return out