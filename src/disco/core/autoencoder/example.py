import numpy as np
import pandas as pd
import torch

from src.disco.core.autoencoder.fit import train_autoencoder
from src.disco.core.autoencoder.artifact import AutoencoderArtifact
from src.disco.core.autoencoder.encoding import AutoencoderRGBInferencer

def build_fake_dataset(
    num_samples: int = 10000,
    num_cell_types: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    probs = rng.dirichlet(
        alpha=np.ones(num_cell_types),
        size=num_samples
    )

    columns = [f"prob_{i}" for i in range(num_cell_types)]
    df = pd.DataFrame(probs, columns=columns)

    return df


def main():
    df = build_fake_dataset(
        num_samples=5000,
        num_cell_types=25
    )

    artifact = train_autoencoder(
        df,
        val_ratio=0.1,
        batch_size=1024,
        in_dim=25,
        bottleneck_dim=3,
        hidden_dim=256,
        lr=1e-4,
        num_epochs=2,
        alpha=0.1,
    )
    save_path = "autoencoder_artifact.pt"
    artifact.save(save_path)
    print(f"Artifact saved to {save_path}")

    loaded_artifact = AutoencoderArtifact.load(save_path)

    inferencer = AutoencoderRGBInferencer(loaded_artifact)


    test_df = df.iloc[:100].copy()
    emb_matrix = torch.tensor(
        test_df[[c for c in test_df.columns if c.startswith("prob_")]].values,
        dtype=torch.float32
    )

    rgb_tensor = inferencer.encode_to_rgb(emb_matrix)

    print("RGB shape:", rgb_tensor.shape)  # [100, 3]

    test_df_with_rgb = inferencer.encode_to_rgb_df(
        test_df,
        emb_matrix
    )

    print(test_df_with_rgb.head())


if __name__ == "__main__":
    main()