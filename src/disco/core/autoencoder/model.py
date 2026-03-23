import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder with symmetric encoder–decoder architecture.

    This model compresses input feature vectors into a low-dimensional latent
    representation (`bottleneck_dim`) and reconstructs them back to the original
    input space. The architecture uses multiple linear layers with ReLU
    activations, LayerNorm, and dropout for regularization.

    The encoder progressively expands and contracts the feature space before
    projecting into the latent bottleneck. The decoder mirrors this structure
    to reconstruct the input.

    Args:
        in_dim (int): Dimensionality of the input features.
        bottleneck_dim (int): Size of the latent representation.
        hidden_dim (int): Base hidden dimension controlling layer widths.
    """

    def __init__(
        self, 
        in_dim: int = 25, 
        bottleneck_dim: int = 3, 
        hidden_dim: int = 512
    ):
        """
        Initialize the autoencoder architecture.

        Args:
            in_dim (int): Number of input features.
            bottleneck_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Base hidden layer size used to construct
                intermediate layers in both encoder and decoder.
        """
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
        """
        Perform a forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - z: Latent representation of shape (batch_size, bottleneck_dim)
                - x_recon: Reconstructed input of shape (batch_size, in_dim)
        """
        z = self.encoder(x)
        return z, self.decoder(z)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features into the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Latent representation of shape
                (batch_size, bottleneck_dim).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations back into input space.

        Args:
            z (torch.Tensor): Latent tensor of shape
                (batch_size, bottleneck_dim).

        Returns:
            torch.Tensor: Reconstructed input of shape (batch_size, in_dim).
        """
        return self.decoder(z)