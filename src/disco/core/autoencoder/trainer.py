from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from disco.core.autoencoder.config import AutoencoderTrainerConfig

from src.disco.core.autoencoder.artifact import AutoencoderArtifact
from src.disco.core.autoencoder.model import Autoencoder


class AutoencoderTrainer:
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        cfg: AutoencoderTrainerConfig,
        device: torch.device | str | None = None,
    ):

        self.df = df
        self._validate_df(df=df)
        self.cfg = cfg
        self.cfg.validate()
        
        self.input_cols = self._get_input_cols(self.df)
        self.in_dim = len(self.input_cols)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.emb_matrix = self._df_to_prob_tensor(self.df)
        dataset = TensorDataset(self.emb_matrix)

        val_size = int(len(dataset) * cfg.val_ratio)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        self.model = Autoencoder(
            in_dim=self.in_dim,
            bottleneck_dim=cfg.bottleneck_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.lr,
        )

        os.makedirs(cfg.save_dir, exist_ok=True)
        
    def _validate_df(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df is empty.")
        
    def _get_input_cols(self, df: pd.DataFrame) -> list[str]:
        if self.cfg.input_cols is not None:
            missing = [col for col in self.cfg.input_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Configured input_cols not found in df: {missing}")
            return list(self.cfg.input_cols)

        input_cols = [col for col in df.columns if col.startswith("prob_")]
        if not input_cols:
            raise ValueError("No probability columns found with prefix 'prob_'.")
        return input_cols

    def _bio_contrastive_loss(
        self,
        z: torch.Tensor,
        orig_probs: torch.Tensor,
        margin: float = 50.0,
        alpha: float = 0.1,
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
        self,
        orig_probs: torch.Tensor,
        pred_logits: torch.Tensor,
        z: torch.Tensor,
        margin: float = 50.0,
        beta: float = 0.1
    ) -> Tuple[torch.Tensor, float, float]:
        pred_log_probs = F.log_softmax(pred_logits, dim=1)
        recon_loss = F.kl_div(pred_log_probs, orig_probs, reduction='batchmean')
        cluster_loss = self._bio_contrastive_loss(z, orig_probs, margin)
        total_loss = recon_loss + beta * cluster_loss
        return total_loss, recon_loss.item(), cluster_loss.item()

    def _df_to_prob_tensor(
        self,
        df: pd.DataFrame
    ) -> torch.Tensor:
        emb_matrix = df[[col for col in df.columns if col.startswith("prob_")]].values # shape = [N, hidden_dim], N = total nodes from all graphs
        emb_matrix_tensor = torch.tensor(emb_matrix, dtype=torch.float32)

        return emb_matrix_tensor

    def _train_one_epoch(
        self,
        model: Module,
        loader: DataLoader,
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
            loss, recon_loss, diversity_loss = self._loss_function(x, out, z, beta=alpha)
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
        self,
        model: Module,
        loader: DataLoader,
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
                loss, recon_loss, diversity_loss = self._loss_function(x, out, z, beta=alpha)
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
        self,
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

    def _save_best_checkpoint(self, save_dir: Path) -> AutoencoderArtifact:
        save_dir.mkdir(exist_ok=True)

        z_min, z_max = self._compute_z_min_max(
            self.model,
            emb_matrix=self.emb_matrix,
            device=self.device,
        )

        artifact = AutoencoderArtifact(
            {k: v.cpu() for k, v in self.model.state_dict().items()},
            input_cols=self.input_cols,
            in_dim=self.in_dim,
            bottleneck_dim=self.cfg.bottleneck_dim,
            hidden_dim=self.cfg.hidden_dim,
            z_min=z_min,
            z_max=z_max
        )

        artifact_path = save_dir / "autoencoder_artifact.pt"
        artifact.save(artifact_path)

        return artifact

    def train(self) -> AutoencoderArtifact:
        for epoch in range(self.cfg.num_epochs):
            self._train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                alpha=self.cfg.alpha,
                epoch=epoch,
                num_epochs=self.cfg.num_epochs,
            )
            self._evaluate_one_epoch(
                model=self.model,
                loader=self.val_loader,
                device=self.device,
                alpha=self.cfg.alpha,
                epoch=epoch,
                num_epochs=self.cfg.num_epochs,
            )

        artifact = self._save_best_checkpoint(self.cfg.save_dir)
        return artifact