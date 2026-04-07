"""Smoke tests: verify model forward passes produce expected output shapes."""
from __future__ import annotations

import pytest
import torch

from core.autoencoder.model import Autoencoder
from core.gcnn.model import GCNClassifier


# ── Autoencoder ────────────────────────────────────────────────────


class TestAutoencoderSmoke:
    def test_forward_pass(self) -> None:
        model = Autoencoder(in_dim=10, bottleneck_dim=3, hidden_dim=64)
        model.eval()

        x = torch.randn(8, 10)
        z, x_recon = model(x)

        assert z.shape == (8, 3)
        assert x_recon.shape == (8, 10)

    def test_encode_decode_roundtrip(self) -> None:
        model = Autoencoder(in_dim=10, bottleneck_dim=3, hidden_dim=64)
        model.eval()

        x = torch.randn(4, 10)
        z = model.encode(x)
        x_hat = model.decode(z)

        assert z.shape == (4, 3)
        assert x_hat.shape == (4, 10)

    def test_single_sample(self) -> None:
        model = Autoencoder(in_dim=5, bottleneck_dim=2, hidden_dim=32)
        model.eval()

        x = torch.randn(1, 5)
        z, x_recon = model(x)

        assert z.shape == (1, 2)
        assert x_recon.shape == (1, 5)


# ── GCNClassifier ─────────────────────────────────────────────────


class TestGCNClassifierSmoke:
    def test_forward_pass(self) -> None:
        model = GCNClassifier(
            in_channels=4,
            hidden_channels=16,
            num_classes=3,
            K=5,
        )
        model.eval()

        num_nodes = 10
        x = torch.randn(num_nodes, 4)
        # Simple chain graph: 0-1-2-...-9
        edge_index = torch.tensor(
            [[i, i + 1] for i in range(num_nodes - 1)]
            + [[i + 1, i] for i in range(num_nodes - 1)],
            dtype=torch.long,
        ).T

        logits = model(x, edge_index)

        assert logits.shape == (num_nodes, 3)

    def test_single_node_no_edges(self) -> None:
        model = GCNClassifier(
            in_channels=2,
            hidden_channels=8,
            num_classes=2,
            K=3,
        )
        model.eval()

        x = torch.randn(1, 2)
        edge_index = torch.zeros(2, 0, dtype=torch.long)

        logits = model(x, edge_index)
        assert logits.shape == (1, 2)
