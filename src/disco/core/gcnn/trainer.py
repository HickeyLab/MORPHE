from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.disco.core.gcnn.artifact import GCNNArtifact
from src.disco.core.gcnn.config import GCNNTrainerConfig
from src.disco.core.gcnn.data import RegionGraphDataset
from src.disco.core.gcnn.model import GCNClassifier


class GCNNTrainer:
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        cfg: GCNNTrainerConfig,
        feature_cols: Sequence[str],
        device: torch.device | str | None = None,
    ):
        cfg.validate()
        self.cfg = cfg

        self.feature_cols = feature_cols
        self._validate_feature_cols(feature_cols)
        
        self.df = df
        self._validate_df(df=df)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _validate_feature_cols(self, feature_cols: Sequence[str]) -> None:
        if not isinstance(feature_cols, (list, tuple)) or len(feature_cols) == 0:
            raise TypeError("feature_cols must be a non-empty sequence of column names.")
        if not all(isinstance(c, str) and c for c in feature_cols):
            raise TypeError("feature_cols must contain only non-empty str column names.")

    def _train_one_epoch(
        self,
        model: Module,
        loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device,
    ) -> float:
        model.train()
        total_loss = 0.0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss / len(loader)

    def _evaluate(
        self,
        model: Module,
        loader: DataLoader,
        device: torch.device,
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
        return correct / total if total else 0.0

    def _validate_df(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df missing feature_cols: {missing}")

        required = [self.cfg.label_col, self.cfg.region_col, *self.cfg.pos_cols]
        missing_req = [c for c in required if c not in df.columns]
        if missing_req:
            raise ValueError(f"df missing required columns: {missing_req}")

    def train(self) -> GCNNArtifact:
        classes_ = sorted(pd.unique(self.df[self.cfg.label_col]).tolist())
        num_classes = len(classes_)

        dataset = RegionGraphDataset(
            df=self.df,
            feature_cols=self.feature_cols,
            label_col=self.cfg.label_col,
            region_col=self.cfg.region_col,
            pos_cols=self.cfg.pos_cols,
            k_neighbors=self.cfg.k_neighbors,
            classes_=classes_,
        )

        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        model = GCNClassifier(
            in_channels=len(self.feature_cols),
            hidden_channels=self.cfg.hidden_channels,
            num_classes=num_classes,
            dropout=float(self.cfg.dropout),
            alpha=float(self.cfg.alpha),
            K=self.cfg.K,
        ).to(self.device)

        optimizer = Adam(
            model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        for epoch in range(self.cfg.epochs):
            loss = self._train_one_epoch(model, train_loader, optimizer, self.device)
            acc = self._evaluate(model, train_loader, self.device)
            print(f"Epoch {epoch}, Loss {loss:.4f}, Train Acc {acc:.4f}")

        return GCNNArtifact(
            {k: v.cpu() for k, v in model.state_dict().items()},
            in_channels=len(self.feature_cols),
            hidden_channels=self.cfg.hidden_channels,
            num_classes=num_classes,
            dropout=float(self.cfg.dropout),
            alpha=float(self.cfg.alpha),
            K=self.cfg.K,
            label_col=self.cfg.label_col,
            region_col=self.cfg.region_col,
            pos_cols=tuple(self.cfg.pos_cols),
            k_neighbors=self.cfg.k_neighbors,
            feature_cols=tuple(self.feature_cols),
            classes_=classes_,
        )