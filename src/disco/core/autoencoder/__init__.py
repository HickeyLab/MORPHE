from src.disco.core.autoencoder.model import Autoencoder
from src.disco.core.autoencoder.artifact import AutoencoderArtifact
from src.disco.core.autoencoder.fit import train_autoencoder
from src.disco.core.autoencoder.encoding import (
    encode_to_rgb,
    add_rgb_to_df,
    AutoencoderRGBInferencer,
)

__all__ = [
    "Autoencoder",
    "AutoencoderArtifact",
    "train_autoencoder",
    "encode_to_rgb",
    "add_rgb_to_df",
    "AutoencoderRGBInferencer",
]