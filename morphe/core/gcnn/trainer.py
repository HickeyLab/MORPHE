from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch_geometric.loader import DataLoader

from ...utils import resolve_device
from .artifact import GCNNArtifact
from .config import GCNNTrainerConfig
from .data import RegionGraphDataset
from .model import GCNClassifier


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
        region_col: str = "unique_region",
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            df: Source dataframe containing features, labels, region identifiers,
                and positional columns.
            cfg: Training configuration.
            feature_cols: Names of dataframe columns to use as node features.
            region_col: Column name identifying distinct regions (each becomes a
                graph). Defaults to ``"unique_region"``. When using
                ``MorpheTrainer``, this is taken from ``PreProcessConfig.region_col``.
            device: Target device for training. If omitted, CUDA is used when
                available; otherwise CPU is used.
        """
        self.cfg = cfg
        self.feature_cols = tuple(feature_cols)
        self.region_col = region_col
        self.df = df
        self.device = resolve_device(device)
        self.seed = seed
        self._validate_feature_cols(self.feature_cols)
        self._validate_df(self.df)

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

        required_cols = [self.cfg.label_col, self.region_col, *self.cfg.pos_cols]
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

    def _set_seed(self, seed: int) -> None:
        """
        Seed Python, NumPy, and PyTorch RNG state for reproducible training.

        Args:
            seed: Random seed to apply.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def train(
        self,
        *,
        save_best: bool = True,
        save_dir: str | Path | None = None,
        save_name: str = "gcnn_artifact.pt",
        verbose: bool = False,
    ) -> GCNNArtifact:
        """
        Train the GCNN model and export it as an artifact.

        Args:
            verbose: Whether to print per-epoch training progress.

        Returns:
            A trained GCNN artifact containing model weights and metadata needed
            to reconstruct the model for inference.
        """
        if self.seed is not None:
            self._set_seed(self.seed)
            if verbose:
                print(f"Using random seed: {self.seed}")

        class_names = tuple(sorted(pd.unique(self.df[self.cfg.label_col]).tolist()))
        num_classes = len(class_names)

        dataset = RegionGraphDataset(
            df=self.df,
            feature_cols=self.feature_cols,
            label_col=self.cfg.label_col,
            region_col=self.region_col,
            pos_cols=self.cfg.pos_cols,
            k_neighbors=self.cfg.k_neighbors,
            classes_=class_names,
        )
        
        # for g in dataset.data_list:
        #     print(
        #         g.region_id,
        #         "nodes=", g.num_nodes,
        #         "edges=", g.edge_index.shape[1],
        #         "feat_dim=", g.x.shape[1],
        #     )

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

        if verbose:
            print(
                "Starting GCNN training "
                f"(device={self.device}, epochs={self.cfg.epochs}, "
                f"graphs={len(dataset)}, batch_size={self.cfg.batch_size}, "
                f"num_classes={num_classes})"
            )

        last_loss = 0.0
        last_acc = 0.0
        best_loss = float("inf")
        best_state_dict: dict | None = None

        for epoch in range(self.cfg.epochs):
            loss = self._train_one_epoch(model, train_loader, optimizer, self.device)
            acc = self._evaluate(model, train_loader, self.device)

            last_loss = loss
            last_acc = acc

            if loss < best_loss:
                best_loss = loss
                best_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{self.cfg.epochs} - "
                    f"loss: {loss:.4f} - train_acc: {acc:.4f}"
                )

        if verbose:
            print(
                "Finished GCNN training "
                f"(final_loss={last_loss:.4f}, final_train_acc={last_acc:.4f})"
            )

        artifact = GCNNArtifact(
            model_state_dict=best_state_dict or {key: value.cpu() for key, value in model.state_dict().items()},
            hidden_channels=self.cfg.hidden_channels,
            num_classes=num_classes,
            dropout=float(self.cfg.dropout),
            alpha=float(self.cfg.alpha),
            K=self.cfg.K,
            label_col=self.cfg.label_col,
            region_col=self.region_col,
            pos_cols=tuple(self.cfg.pos_cols),
            k_neighbors=self.cfg.k_neighbors,
            feature_cols=tuple(self.feature_cols),
            classes_=class_names,
        )

        if save_best:
            out_dir = Path(save_dir) if save_dir is not None else Path(".")
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact.save(out_dir / save_name)
            if verbose:
                print(f"Saved best GCNN artifact to {out_dir / save_name}")

        return artifact