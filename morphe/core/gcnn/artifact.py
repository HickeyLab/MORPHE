from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from utils import resolve_device, resolve_dtype
from core.gcnn.model import GCNClassifier


@dataclass(frozen=True, slots=True)
class GCNNArtifact:
    """
    Serializable artifact for reconstructing a trained GCNN classifier.

    This artifact stores the trained model weights together with the
    architecture parameters and metadata required to rebuild the model for
    inference.
    """

    model_state_dict: Mapping[str, torch.Tensor]
    hidden_channels: int
    num_classes: int
    dropout: float
    alpha: float
    K: int
    k_neighbors: int
    label_col: str
    feature_cols: tuple[str, ...]
    region_col: str
    pos_cols: tuple[str, ...]
    classes_: tuple[str, ...]

    def build_model(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> GCNClassifier:
        """
        Reconstruct the trained GCNN model from the artifact.

        Args:
            device: Target device for the reconstructed model. If omitted,
                CUDA is used when available; otherwise CPU is used.

        Returns:
            A GCNN classifier loaded with the artifact weights and placed in
            evaluation mode.
        """
        model = GCNClassifier(
            in_channels=len(self.feature_cols),
            hidden_channels=self.hidden_channels,
            num_classes=self.num_classes,
            dropout=self.dropout,
            alpha=self.alpha,
            K=self.K,
        )

        model.load_state_dict(dict(self.model_state_dict), strict=True)
        
        device = resolve_device(device)
        dtype = resolve_dtype(device, dtype)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        return model

    def save(self, path: str | Path) -> None:
        """
        Save the artifact to disk.

        The model weights are moved to CPU before serialization so the saved
        artifact is device-agnostic.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cpu_state = {
            key: value.detach().cpu()
            for key, value in self.model_state_dict.items()
        }

        payload = {
            "model_state_dict": cpu_state,
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
            "classes_": tuple(self.classes_),
        }

        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> GCNNArtifact:
        """
        Load an artifact from disk.

        Args:
            path: Path to a serialized GCNN artifact file.

        Returns:
            A reconstructed ``GCNNArtifact``.

        Raises:
            TypeError: If the loaded payload is not mapping-like.
            ValueError: If required keys are missing from the payload.
        """
        path = Path(path)

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Invalid GCNNArtifact file: expected mapping payload, got {type(payload)}"
            )

        required_keys = (
            "model_state_dict",
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
            "classes_",
        )

        for key in required_keys:
            if key not in payload:
                raise ValueError(f"Invalid GCNNArtifact file: missing key '{key}'")

        return cls(
            model_state_dict=payload["model_state_dict"],
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
            classes_=tuple(payload["classes_"]),
        )