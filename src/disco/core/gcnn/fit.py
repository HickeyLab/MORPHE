from typing import Iterable, Sequence
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim import Optimizer, Adam
from src.disco.core.gcnn.artifact import GCNNArtifact

from src.disco.core.gcnn.data import RegionGraphDataset
from src.disco.core.gcnn.model import GCNClassifier


def _train_one_epoch(
        model: Module,
        loader: Iterable[Data],
        optimizer: Optimizer,
        device: torch.device
) -> float:
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _evaluate(
        model: Module,
        loader: Iterable[Data],
        device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total


def _validate_train_gcnn_args(
        *,
        hidden_channels: int,
        dropout: float,
        alpha: float,
        K: int,
        lr: float,
        weight_decay: float,
        epochs: int,
) -> None:
    if hidden_channels < 1:
        raise ValueError("hidden_channels must be >= 1.")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("dropout must be in [0, 1).")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    if K < 1:
        raise ValueError("K must be >= 1.")
    if lr <= 0:
        raise ValueError("lr must be > 0.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")


def train_gcnn(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        label_col: str = "Cell Type",
        region_col: str = "unique_region",
        pos_cols: Sequence[str] = ("x", "y"),
        k_neighbors: int = 20,
        hidden_channels: int = 768,
        dropout: float = 0.1,
        alpha: float = 0.9,
        K: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 40,
        device: torch.device | str | None = None
) -> GCNNArtifact:
    # Validate GCNN arguments not in dataset validation
    _validate_train_gcnn_args(
        hidden_channels=hidden_channels,
        dropout=dropout,
        alpha=alpha,
        K=K,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs
    )

    # Create ordered classes for reproducibility
    classes_ = sorted(pd.unique(df[label_col]).tolist())
    num_classes = len(classes_)

    # Create dataset using radius-graph version
    dataset = RegionGraphDataset(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        region_col=region_col,
        pos_cols=pos_cols,
        k_neighbors=k_neighbors,
        classes_=classes_
    )

    # Graph-level batching: each batch = one region graph
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Select device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Initialize GNN model
    model = GCNClassifier(
        in_channels=len(feature_cols),
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=dropout,
        alpha=alpha,
        K=K
    ).to(device)

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        # Train for one epoch
        loss = _train_one_epoch(model, train_loader, optimizer, device)
        # Evaluate on training set (node-level accuracy)
        acc = _evaluate(model, train_loader, device)
        print(f"Epoch {epoch}, Loss {loss:.4f}, Train Acc {acc:.4f}")

    return GCNNArtifact(
        {k: v.cpu() for k, v in model.state_dict().items()},
        in_channels=len(feature_cols),
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=dropout,
        alpha=alpha,
        K=K,
        label_col=label_col,
        region_col=region_col,
        pos_cols=list(pos_cols),
        k_neighbors=k_neighbors,
        feature_cols=list(feature_cols),
        classes_=classes_,
    )
