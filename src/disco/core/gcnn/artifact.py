from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Tuple
import torch

from disco.core.gcnn.model import GCNClassifier


@dataclass(frozen=True)
class GCNNArtifact:
    state_dict: Mapping[str, torch.Tensor]
    in_channels: int
    hidden_channels: int
    num_classes: int
    dropout: float
    alpha: float
    K: int
    k_neighbors: int
    label_col: str
    feature_cols: Tuple[str, ...]
    region_col: str
    pos_cols: Tuple[str, ...]
    classes_: List[str]

    def build_model(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> GCNClassifier:
        model = GCNClassifier(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_classes=self.num_classes,
            dropout=self.dropout,
            alpha=self.alpha,
            K=self.K,
        )
        model.load_state_dict(self.state_dict, strict=True)
        
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        if device is not None:
            model = model.to(device)

        model.eval()
        return model

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict,
                "in_channels": self.in_channels,
                "hidden_channels": self.hidden_channels,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "alpha": self.alpha,
                "K": self.K,
                "k_neighbors": self.k_neighbors,
                "label_col": self.label_col,
                "feature_cols": self.feature_cols,
                "region_col": self.region_col,
                "pos_cols": self.pos_cols,
            },
            str(path),
        )


    @staticmethod
    def load(path: str | Path) -> "GCNNArtifact":
        payload = torch.load(path, map_location="cpu")

        return GCNNArtifact(
            state_dict=payload["state_dict"],
            in_channels=payload["in_channels"],
            hidden_channels=payload["hidden_channels"],
            num_classes=payload["num_classes"],
            dropout=payload["dropout"],
            alpha=payload["alpha"],
            K=payload["K"],
            k_neighbors=payload["k_neighbors"],
            label_col=payload["label_col"],
            feature_cols=tuple(payload["feature_cols"]),
            region_col=payload["region_col"],
            pos_cols=tuple(payload["pos_cols"]),
        )
