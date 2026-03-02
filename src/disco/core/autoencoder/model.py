import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(
        self, 
        in_dim: int = 25, 
        bottleneck_dim: int = 3, 
        hidden_dim: int = 512
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim/4)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/4)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/4), int(hidden_dim/2)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/2), int(hidden_dim)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/4)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/4), bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, int(hidden_dim/4)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/4)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/4), int(hidden_dim/2)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/2), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/2)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.ReLU(),
            nn.LayerNorm(int(hidden_dim/4)),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim/4), in_dim)
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, self.decoder(z)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)