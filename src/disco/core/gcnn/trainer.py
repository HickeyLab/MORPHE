from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch_geometric.loader import DataLoader

from src.disco.core.gcnn.artifact import GCNNArtifact
from src.disco.core.gcnn.config import GCNNTrainerConfig
from src.disco.core.gcnn.data import RegionGraphDataset
from src.disco.core.gcnn.model import GCNClassifier


class GCNNTrainer:
    """
    Train a graph convolutional classifier from a tabular regional dataset.

    This trainer validates the input dataframe and feature-column selection,
    constructs the graph dataset and model, runs supervised training, and
    returns a serialized artifact containing the trained model weights and
    reconstruction metadata needed for inference.
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        cfg: GCNNTrainerConfig,
        feature_cols: Sequence[str],
        device: torch.device | str | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            df: Source dataframe containing features, labels, region identifiers,
                and positional columns.
            cfg: Training configuration.
            feature_cols: Names of dataframe columns to use as node features.
            device: Target device for training. If omitted, CUDA is used when
                available; otherwise CPU is used.
        """
        self.cfg = cfg
        self.feature_cols = tuple(feature_cols)
        self.df = df
        self.device = self._resolve_device(device)

        self._validate_feature_cols(self.feature_cols)
        self._validate_df(self.df)

    @staticmethod
    def _resolve_device(device: torch.device | str | None) -> torch.device:
        """
        Resolve the training device.

        Args:
            device: User-supplied device specification.

        Returns:
            A normalized torch device.
        """
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _validate_feature_cols(self, feature_cols: Sequence[str]) -> None:
        """
        Validate the feature-column specification.

        Args:
            feature_cols: Candidate feature-column names.

        Raises:
            TypeError: If the feature-column sequence is empty or contains
                invalid entries.
        """
        if len(feature_cols) == 0:
            raise TypeError("feature_cols must be a non-empty sequence of column names.")
        if not all(isinstance(col, str) and col for col in feature_cols):
            raise TypeError("feature_cols must contain only non-empty str column names.")

    def _validate_df(self, df: pd.DataFrame) -> None:
        """
        Validate the source dataframe against trainer requirements.

        Args:
            df: Input dataframe.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If required columns are missing.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")

        missing_feature_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_feature_cols:
            raise ValueError(f"df missing feature_cols: {missing_feature_cols}")

        required_cols = [self.cfg.label_col, self.cfg.region_col, *self.cfg.pos_cols]
        missing_required_cols = [col for col in required_cols if col not in df.columns]
        if missing_required_cols:
            raise ValueError(f"df missing required columns: {missing_required_cols}")

    def _train_one_epoch(
        self,
        model: GCNClassifier,
        loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device,
    ) -> float:
        """
        Train the model for one epoch.

        Args:
            model: Graph classifier to optimize.
            loader: Training dataloader.
            optimizer: Optimizer for parameter updates.
            device: Device on which computation should occur.

        Returns:
            Mean training loss across the epoch.
        """
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
        model: GCNClassifier,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """
        Evaluate classification accuracy over a dataloader.

        Args:
            model: Graph classifier to evaluate.
            loader: Evaluation dataloader.
            device: Device on which computation should occur.

        Returns:
            Accuracy over all examples in the loader.
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum().item())
                total += int(data.y.size(0))

        return correct / total if total else 0.0

    def train(self) -> GCNNArtifact:
        """
        Train the GCNN model and export it as an artifact.

        Returns:
            A trained GCNN artifact containing model weights and metadata needed
            to reconstruct the model for inference.
        """
        class_names = tuple(sorted(pd.unique(self.df[self.cfg.label_col]).tolist()))
        num_classes = len(class_names)

        dataset = RegionGraphDataset(
            df=self.df,
            feature_cols=self.feature_cols,
            label_col=self.cfg.label_col,
            region_col=self.cfg.region_col,
            pos_cols=self.cfg.pos_cols,
            k_neighbors=self.cfg.k_neighbors,
            classes_=class_names,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

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
            model_state_dict={key: value.cpu() for key, value in model.state_dict().items()},
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
            classes_=class_names,
        )