from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Tuple
import torch

from src.disco.core.gcnn.model import GCNClassifier


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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model = model.to(device)
        model.eval()
        return model


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cpu_state = {k: v.detach().cpu() for k, v in self.state_dict.items()}

        payload = {
            "state_dict": cpu_state,
            "in_channels": int(self.in_channels),
            "hidden_channels": int(self.hidden_channels),
            "num_classes": int(self.num_classes),
            "dropout": float(self.dropout),
            "alpha": float(self.alpha),
            "K": int(self.K),
            "k_neighbors": int(self.k_neighbors),
            "label_col": str(self.label_col),
            "feature_cols": tuple(self.feature_cols),
            "region_col": str(self.region_col),
            "pos_cols": tuple(self.pos_cols),
            "classes_": list(self.classes_),
        }

        torch.save(payload, str(path))

    @staticmethod
    def load(path: str | Path) -> "GCNNArtifact":
        path = Path(path)

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")

        required_keys = (
            "state_dict",
            "in_channels",
            "hidden_channels",
            "num_classes",
            "dropout",
            "alpha",
            "K",
            "k_neighbors",
            "label_col",
            "feature_cols",
            "region_col",
            "pos_cols",
            "classes_",   # ✅ 现在要求存在
        )

        for k in required_keys:
            if k not in payload:
                raise ValueError(f"Invalid GCNNArtifact file: missing key '{k}'")

        return GCNNArtifact(
            state_dict=payload["state_dict"],
            in_channels=int(payload["in_channels"]),
            hidden_channels=int(payload["hidden_channels"]),
            num_classes=int(payload["num_classes"]),
            dropout=float(payload["dropout"]),
            alpha=float(payload["alpha"]),
            K=int(payload["K"]),
            k_neighbors=int(payload["k_neighbors"]),
            label_col=str(payload["label_col"]),
            feature_cols=tuple(payload["feature_cols"]),
            region_col=str(payload["region_col"]),
            pos_cols=tuple(payload["pos_cols"]),
            classes_=list(payload["classes_"]),
        )