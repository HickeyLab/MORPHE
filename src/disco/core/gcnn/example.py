import numpy as np
import pandas as pd

from src.disco.core.gcnn import train_gcnn, GCNNInferencer, GCNNArtifact


def build_fake_dataset(
    num_regions: int = 5,
    cells_per_region: int = 200,
    num_features: int = 16,
    num_classes: int = 4,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    rows = []

    for r in range(num_regions):
        region_id = f"region_{r}"

        x = rng.uniform(0, 1000, size=cells_per_region)
        y = rng.uniform(0, 1000, size=cells_per_region)

        features = rng.normal(size=(cells_per_region, num_features))

        labels = rng.integers(0, num_classes, size=cells_per_region)
        labels = [f"class_{l}" for l in labels]

        for i in range(cells_per_region):
            row = {
                "unique_region": region_id,
                "x": x[i],
                "y": y[i],
                "Cell Type": labels[i],
            }

            for f in range(num_features):
                row[f"feat_{f}"] = features[i, f]

            rows.append(row)

    return pd.DataFrame(rows)


def main():

    df = build_fake_dataset()

    feature_cols = [c for c in df.columns if c.startswith("feat_")]

    # ------------------------
    # Train
    # ------------------------
    artifact = train_gcnn(
        df=df,
        feature_cols=feature_cols,
        label_col="Cell Type",
        epochs=2,
        hidden_channels=128,
    )

    artifact.save("gcnn_model.pt")
    print("Model saved.")
    artifact1 = GCNNArtifact.load("gcnn_model.pt")
    print("Model loaded.")

    # ------------------------
    # Inference
    # ------------------------
    inferencer = GCNNInferencer(artifact1)

    result_df = inferencer.predict_proba(df)

    print(result_df.head())


if __name__ == "__main__":
    main()