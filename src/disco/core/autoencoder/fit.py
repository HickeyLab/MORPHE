from typing import Iterable, Tuple
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from disco.core.autoencoder.artifact import AutoencoderArtifact

from disco.core.autoencoder.model import Autoencoder

def _bio_contrastive_loss(
    z: torch.Tensor, 
    orig_probs: torch.Tensor, 
    margin: float = 50.0,
    alpha: float = 0.1
) -> torch.Tensor:
    """
    Biologically weighted contrastive loss.

    Goal:
      - Keep samples of the same cell type close together (intra-class cohesion)
      - Push different cell types apart (inter-class separation)
      - Allow biologically similar cell types (based on probability similarity)
        to be closer in the latent space

    Args:
        z (torch.Tensor): Latent embeddings of shape [B, D]
        orig_probs (torch.Tensor): Cell type probability distributions [B, C]
        margin (float): Margin distance for inter-class separation
        alpha (float): Strength of biological similarity weighting
    """
    B = z.size(0)
    device = z.device

    # --- Step 1: Compute pairwise Euclidean distances in latent space ---
    dists = torch.cdist(z, z, p=2)  # [B, B]

    # --- Step 2: Identify same-type and different-type pairs ---
    labels = orig_probs.argmax(dim=1)               # Hard label per sample
    same = (labels.unsqueeze(1) == labels.unsqueeze(0))  # [B, B]
    diff = ~same

    # --- Step 3: Compute biological similarity between cells ---
    # Use cosine similarity between cell-type probability distributions
    # to reflect biological closeness between different types
    prob_sim = F.cosine_similarity(
        orig_probs.unsqueeze(1),  # [B, 1, C]
        orig_probs.unsqueeze(0),  # [1, B, C]
        dim=-1
    )  # [B, B]
    prob_sim = torch.clamp(prob_sim, 0, 1)  # Ensure range [0, 1]

    # --- Step 4: Intra-class loss (same type) ---
    # Encourage embeddings of the same cell type to be close together
    intra_loss = dists[same].sum() / (same.sum() + 1e-8)

    # --- Step 5: Inter-class loss (different types) ---
    # Encourage different types to be far apart,
    # but scale the penalty by biological similarity
    # (more similar types get a smaller penalty)
    inter_weight = (1 - alpha * prob_sim[diff])
    inter_loss = (inter_weight * torch.clamp(margin - dists[diff], min=0)).sum() / (diff.sum() + 1e-8)

    # --- Step 6: Combine total loss ---
    total_loss = 0.1 * intra_loss + inter_loss

    return total_loss

# TODO: ASK ABOUT BETA AND ALPHA HERE (_bio_contrastive_loss)
def _loss_function(
    orig_probs: torch.Tensor, 
    pred_logits: torch.Tensor, 
    z: torch.Tensor, 
    margin: float = 50.0, 
    beta: float = 0.1
) -> Tuple[torch.Tensor, float, float]:
    pred_log_probs = F.log_softmax(pred_logits, dim=1)
    recon_loss = F.kl_div(pred_log_probs, orig_probs, reduction='batchmean')
    cluster_loss = _bio_contrastive_loss(z, orig_probs, margin)
    total_loss = recon_loss + beta * cluster_loss
    return total_loss, recon_loss.item(), cluster_loss.item()

def _df_to_prob_tensor(
    df: pd.DataFrame
) -> torch.Tensor:
    emb_matrix = df[[col for col in df.columns if col.startswith("prob_")]].values # shape = [N, hidden_dim], N = total nodes from all graphs
    emb_matrix_tensor = torch.tensor(emb_matrix, dtype=torch.float32)
    
    return emb_matrix_tensor

def _train_one_epoch(
    model: Module, 
    loader: Iterable, 
    optimizer: Optimizer,
    device: torch.device,
    alpha: float,
    epoch: int,
    num_epochs: int,
) -> None:
    model.train()
    total_loss = 0
    total_recon = 0
    total_div = 0
    t_correct = 0
    t_total = 0
    for batch in tqdm(loader, desc=f"Train Epoch {epoch+1}"):
        x = batch[0].to(device)
        optimizer.zero_grad()
        z, out = model(x)
        loss, recon_loss, diversity_loss = _loss_function(x, out, z, beta=alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss
        total_div += diversity_loss
        out_label = out.argmax(dim=1)
        x_label = x.argmax(dim=1)
        t_correct += (out_label == x_label).sum().item()
        t_total += x.size(0)
    val_acc = t_correct / t_total
    total_loss /= len(loader)
    total_recon /= len(loader)
    total_div /= len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss:.4f} | Recon: {total_recon:.4f} | Diversity: {total_div:.4f} | Train Acc: {val_acc:.4f}")
    
def _evaluate_one_epoch(
    model: Module, 
    loader: Iterable, 
    device: torch.device,
    alpha: float,
    epoch: int,
    num_epochs: int
) -> None:
    model.eval()
    val_loss = 0
    val_recon = 0
    val_div = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val Epoch {epoch+1}"):
            x = batch[0].to(device)
            z, out = model(x)
            loss, recon_loss, diversity_loss = _loss_function(x, out, z, beta=alpha)
            val_loss += loss.item()
            val_recon += recon_loss
            val_div += diversity_loss
            x_label = x.argmax(dim=1)
            out_label = out.argmax(dim=1)
            val_correct += (out_label == x_label).sum().item()
            val_total += x.size(0)
    val_acc = val_correct / val_total
    val_loss /= len(loader)
    val_recon /= len(loader)
    val_div /= len(loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | Diversity: {val_div:.4f} | Val Acc: {val_acc:.4f}")
    

def _compute_z_min_max(
    model: Autoencoder,
    emb_matrix: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        z = model.encoder(emb_matrix.to(device))
        z_min = z.min(dim=0).values.detach().cpu()
        z_max = z.max(dim=0).values.detach().cpu()
    return z_min, z_max

def _validate_train_autoencoder_args(
    df: pd.DataFrame,
    *,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int,
    lr: float,
    num_epochs: int,
    alpha: float,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty.")

    if not isinstance(val_ratio, (int, float)):
        raise TypeError("val_ratio must be a float in [0, 1).")
    val_ratio = float(val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1).")

    for name, val in (
        ("batch_size", batch_size),
        ("num_workers", num_workers),
        ("in_dim", in_dim),
        ("bottleneck_dim", bottleneck_dim),
        ("hidden_dim", hidden_dim),
        ("num_epochs", num_epochs),
    ):
        if not isinstance(val, int):
            raise TypeError(f"{name} must be an int.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if in_dim <= 0:
        raise ValueError("in_dim must be > 0.")
    if bottleneck_dim <= 0:
        raise ValueError("bottleneck_dim must be > 0.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be > 0.")
    if bottleneck_dim >= in_dim:
        raise ValueError("bottleneck_dim must be < in_dim.")

    if not isinstance(lr, (int, float)):
        raise TypeError("lr must be a float > 0.")
    lr = float(lr)
    if not (lr > 0.0):
        raise ValueError("lr must be > 0.")

    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a float >= 0.")
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be >= 0.")

def train_autoencoder(
    df: pd.DataFrame,
    *,
    val_ratio: float = 0.1,
    batch_size: int = 4096,
    num_workers: int = 4,
    in_dim: int = 25,
    bottleneck_dim: int = 3,
    hidden_dim: int = 512,
    lr: float = 1e-6,
    num_epochs: int = 20,
    alpha: float = 0.1,
    device: torch.device | None = None
):
    _validate_train_autoencoder_args(
        df=df,
        val_ratio=val_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        in_dim=in_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        num_epochs=num_epochs,
        alpha=alpha
    )
    
    emb_matrix = _df_to_prob_tensor(df)
    dataset = TensorDataset(emb_matrix)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Select device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    # Initialize Autoencoder model
    model = Autoencoder(
        in_dim=in_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=lr
    )

    for epoch in range(num_epochs):
        _train_one_epoch(
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            device=device, 
            alpha=alpha,
            epoch=epoch,
            num_epochs=num_epochs
        )
        _evaluate_one_epoch(
            model=model, 
            loader=val_loader, 
            device=device, 
            alpha=alpha,
            epoch=epoch,
            num_epochs=num_epochs
        )
    
    z_min, z_max = _compute_z_min_max(model, emb_matrix=emb_matrix, device=device)
    
    return AutoencoderArtifact(
        {k: v.cpu() for k, v in model.state_dict().items()},
        in_dim=in_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        z_min=z_min,
        z_max=z_max
    )