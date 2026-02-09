from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from disco.core.gcnn.artifact import GCNNArtifact
from disco.core.gcnn.data import RegionGraphDataset


def gcnn_predict_proba(
    artifact: GCNNArtifact,
    df: pd.DataFrame,
    output_file_path: Path,
    *,
    device: torch.device | str | None = None,
):
    # Validate fields
    if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty.")
    if not output_file_path:
        raise ValueError("Output file path empty.")
    
    # Load and build model
    model = artifact.build_model(device=device)
    model_device = next(model.parameters()).device
    
    dataset = RegionGraphDataset(
        df=df,
        feature_cols=artifact.feature_cols,
        label_col=None,
        region_col=artifact.region_col,
        pos_cols=artifact.pos_cols,
        k_neighbors=artifact.k_neighbors,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    all_probs = []
    all_rows = []
    all_x = []
    
    with torch.no_grad():
        for batch in loader:
            # Each batch is ONE region graph
            batch = batch.to(model_device)

            # GNN forward pass
            out = model(batch.x, batch.edge_index)  # [num_nodes, num_classes]

            # softmax probability for each node
            probs = torch.softmax(out, dim=1).cpu().numpy()

            # store
            all_probs.append(probs)
            all_rows.append(batch.row_idx.cpu().numpy())
            all_x.append(batch.x.cpu().numpy())    # original features not needed usually
            
    # ----------------------------------------
    # Concatenate results across all regions
    # ----------------------------------------

    all_probs = np.concatenate(all_probs, axis=0)   # shape: [num_cells_total, num_classes]
    all_rows = np.concatenate(all_rows, axis=0)

    x_col, y_col = artifact.pos_cols[0], artifact.pos_cols[1]
    r_col = artifact.region_col

    result_df = df.loc[all_rows, [x_col, y_col, r_col]].copy().reset_index(drop=True)

    # Add prob columns
    for i in range(all_probs.shape[1]):
        result_df[f"prob_class{i}"] = all_probs[:, i]

    # Order columns
    cols = [x_col, y_col, r_col] + [f"prob_class{i}" for i in range(all_probs.shape[1])]
    result_df = result_df[cols]

    # Save
    result_df.to_csv(output_file_path, index=False)