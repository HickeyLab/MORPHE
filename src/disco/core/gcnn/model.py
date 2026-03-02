from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GraphNorm


class GCNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        *,
        dropout: float = 0.1,
        alpha: float = 0.9,      # teleport probability → higher = stronger self-feature weight
        K: int = 20            # number of propagation steps
    ):
        super().__init__()
        
        # Validate bounds on fields
        if in_channels <= 0 or hidden_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels/hidden_channels/num_classes must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        if K <= 0:
            raise ValueError("K must be positive.")

        # MLP before propagation (APPNP standard)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)

        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)

        # APPNP propagation module
        self.propagation = APPNP(
            K=K,
            alpha=alpha,
            dropout=dropout
        )

        self.dropout = dropout
        self.out_lin = nn.Linear(hidden_channels, num_classes)

    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor
    ) -> Tensor:
        # ----- MLP encoder -----
        x = self.lin1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- APPNP propagation (self-feature preserving) -----
        x = self.propagation(x, edge_index)

        # ----- Classification head -----
        x = self.out_lin(x)
        return x