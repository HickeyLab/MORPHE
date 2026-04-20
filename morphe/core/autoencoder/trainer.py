from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from .config import AutoencoderTrainerConfig
from ...utils import resolve_device
from .artifact import AutoencoderArtifact
from .model import Autoencoder


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """
    Container for aggregated metrics from a single training or validation epoch.

    Attributes:
        loss: Mean total loss across the epoch.
        recon_loss: Mean reconstruction loss across the epoch.
        cluster_loss: Mean clustering or contrastive loss across the epoch.
        acc: Mean accuracy across the epoch.
    """
    loss: float
    recon_loss: float
    cluster_loss: float
    acc: float


class AutoencoderTrainer:
    """
    Trainer for fitting an autoencoder on probability-vector inputs.

    This trainer validates the input dataframe, extracts the configured input
    columns, constructs train/validation splits, optimizes the autoencoder, and
    returns the best-performing model as an artifact.
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        cfg: AutoencoderTrainerConfig,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the autoencoder trainer.

        Args:
            df: Input dataframe containing the probability features used for
                training.
            cfg: Training configuration for the autoencoder.
            device: Device on which to train. If omitted, a default device is
                resolved automatically.
        """
        self.df = df
        self.cfg = cfg
        self._validate_df(df=df)

        self.input_cols = self._get_input_cols(self.df)
        self.in_dim = len(self.input_cols)

        self.device = resolve_device(device)
        self.seed = seed

        self.emb_matrix = self._df_to_prob_tensor(self.df)

        self.model: Autoencoder | None = None
        self.optimizer: Optimizer | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None

    def _validate_df(self, df: pd.DataFrame) -> None:
        """
        Validate the training dataframe and required input columns.

        Args:
            df: Dataframe to validate.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If the dataframe is empty, required columns are missing,
                or the selected input columns contain null values.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("df is empty.")

        input_cols = self.cfg.input_cols
        if input_cols is not None:
            missing = [col for col in input_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Configured input_cols not found in df: {missing}")
            cols_to_check = list(input_cols)
        else:
            cols_to_check = [col for col in df.columns if col.startswith("prob_")]
            if not cols_to_check:
                raise ValueError("No probability columns found with prefix 'prob_'.")

        if df[cols_to_check].isnull().any().any():
            raise ValueError("Input probability columns contain null values.")

    def _get_input_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Determine which dataframe columns should be used as model inputs.

        Args:
            df: Input dataframe.

        Returns:
            The list of input column names. Uses configured ``input_cols`` when
            present; otherwise falls back to columns prefixed with ``prob_``.
        """
        if self.cfg.input_cols is not None:
            return list(self.cfg.input_cols)

        return [col for col in df.columns if col.startswith("prob_")]

    def _set_seed(self, seed: int) -> None:
        """
        Seed Python, NumPy, and PyTorch RNGs for reproducible training behavior.

        Args:
            seed: Random seed to apply.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _build_loaders(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Build train and validation dataloaders from the embedded probability data.

        Args:
            seed: Optional seed used to make the dataset split and shuffled
                training loader reproducible.

        Returns:
            A tuple of ``(train_loader, val_loader)``.
        """
        dataset = TensorDataset(self.emb_matrix)

        val_size = int(len(dataset) * self.cfg.val_ratio)
        train_size = len(dataset) - val_size

        split_generator = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else None
        )

        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=split_generator,
        )

        train_loader_generator = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else None
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            generator=train_loader_generator,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        return train_loader, val_loader

    def _build_model_and_optimizer(self) -> tuple[Autoencoder, Optimizer]:
        """
        Construct the autoencoder model and its optimizer.

        Returns:
            A tuple containing the initialized model and optimizer.
        """
        model = Autoencoder(
            in_dim=self.in_dim,
            bottleneck_dim=self.cfg.bottleneck_dim,
            hidden_dim=self.cfg.hidden_dim,
        ).to(self.device)

        optimizer = Adam(
            model.parameters(),
            lr=self.cfg.lr,
        )

        return model, optimizer

    def _bio_contrastive_loss(
        self,
        z: torch.Tensor,
        orig_probs: torch.Tensor,
        margin: float = 50.0,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute a biologically motivated contrastive loss in latent space.

        This loss encourages embeddings with the same dominant class to remain
        close while pushing embeddings with different dominant classes apart,
        modulated by cosine similarity between the original probability vectors.

        Args:
            z: Latent representations produced by the encoder.
            orig_probs: Original probability vectors for the batch.
            margin: Margin used for separating dissimilar examples.
            alpha: Scaling factor applied to the similarity-based inter-class
                weighting.

        Returns:
            The scalar contrastive loss for the batch.
        """
        dists = torch.cdist(z, z, p=2)

        labels = orig_probs.argmax(dim=1)
        same = labels.unsqueeze(1) == labels.unsqueeze(0)
        diff = ~same

        prob_sim = F.cosine_similarity(
            orig_probs.unsqueeze(1),
            orig_probs.unsqueeze(0),
            dim=-1,
        )
        prob_sim = torch.clamp(prob_sim, 0, 1)

        intra_loss = dists[same].sum() / (same.sum() + 1e-8)

        inter_weight = 1 - alpha * prob_sim[diff]
        inter_loss = (
            inter_weight * torch.clamp(margin - dists[diff], min=0)
        ).sum() / (diff.sum() + 1e-8)

        return 0.1 * intra_loss + inter_loss

    def _loss_function(
        self,
        orig_probs: torch.Tensor,
        pred_logits: torch.Tensor,
        z: torch.Tensor,
        margin: float = 50.0,
        beta: float = 0.1,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Compute the full training loss for a batch.

        The total loss combines KL-divergence reconstruction loss with the
        latent-space contrastive loss.

        Args:
            orig_probs: Ground-truth probability vectors.
            pred_logits: Decoder output logits.
            z: Latent representations for the batch.
            margin: Margin used in the contrastive loss.
            beta: Weight applied to the contrastive component.

        Returns:
            A tuple containing:
                - total loss tensor
                - reconstruction loss as a float
                - contrastive loss as a float
        """
        pred_log_probs = F.log_softmax(pred_logits, dim=1)
        recon_loss = F.kl_div(pred_log_probs, orig_probs, reduction="batchmean")
        cluster_loss = self._bio_contrastive_loss(z, orig_probs, margin)
        total_loss = recon_loss + beta * cluster_loss
        return total_loss, recon_loss.item(), cluster_loss.item()

    def _df_to_prob_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Convert the selected dataframe input columns into a float tensor.

        Args:
            df: Input dataframe.

        Returns:
            A tensor containing the selected probability features.
        """
        emb_matrix = df.loc[:, self.input_cols].to_numpy()
        return torch.tensor(emb_matrix, dtype=torch.float32)

    def _train_one_epoch(
        self,
        model: Autoencoder,
        loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device,
        alpha: float,
        *,
        verbose: bool = False,
    ) -> EpochMetrics:
        """
        Train the model for one epoch.

        Args:
            model: Autoencoder to optimize.
            loader: Training dataloader.
            optimizer: Optimizer used for gradient updates.
            device: Device on which training is performed.
            alpha: Weighting term passed into the batch loss computation.
            verbose: Whether to wrap the loader with a progress bar.

        Returns:
            Aggregated metrics for the epoch.
        """
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_div = 0.0
        correct = 0
        total = 0

        iterator = tqdm(loader, desc="Train", leave=False) if verbose else loader
        for batch in iterator:
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
            correct += (out_label == x_label).sum().item()
            total += x.size(0)

        return EpochMetrics(
            loss=total_loss / len(loader),
            recon_loss=total_recon / len(loader),
            cluster_loss=total_div / len(loader),
            acc=correct / total,
        )

    def _evaluate_one_epoch(
        self,
        model: Autoencoder,
        loader: DataLoader,
        device: torch.device,
        alpha: float,
        *,
        verbose: bool = False,
    ) -> EpochMetrics:
        """
        Evaluate the model for one epoch without gradient updates.

        Args:
            model: Autoencoder to evaluate.
            loader: Validation dataloader.
            device: Device on which evaluation is performed.
            alpha: Weighting term passed into the batch loss computation.
            verbose: Whether to wrap the loader with a progress bar.

        Returns:
            Aggregated validation metrics for the epoch.
        """
        model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_div = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            iterator = tqdm(loader, desc="Val", leave=False) if verbose else loader
            for batch in iterator:
                x = batch[0].to(device)
                z, out = model(x)
                loss, recon_loss, diversity_loss = self._loss_function(x, out, z, beta=alpha)

                total_loss += loss.item()
                total_recon += recon_loss
                total_div += diversity_loss
                x_label = x.argmax(dim=1)
                out_label = out.argmax(dim=1)
                correct += (out_label == x_label).sum().item()
                total += x.size(0)

        return EpochMetrics(
            loss=total_loss / len(loader),
            recon_loss=total_recon / len(loader),
            cluster_loss=total_div / len(loader),
            acc=correct / total,
        )

    def _compute_z_min_max(
        self,
        model: Autoencoder,
        emb_matrix: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-dimension latent minima and maxima over the full dataset.

        Args:
            model: Trained autoencoder model.
            emb_matrix: Full embedded input matrix.
            device: Device on which to run the encoder.

        Returns:
            A tuple of ``(z_min, z_max)`` tensors for the latent dimensions.
        """
        model.eval()
        with torch.no_grad():
            z = model.encoder(emb_matrix.to(device))
            z_min = z.min(dim=0).values.detach().cpu()
            z_max = z.max(dim=0).values.detach().cpu()
        return z_min, z_max

    def _build_best_artifact(self) -> AutoencoderArtifact:
        """
        Build an artifact from the current best model state without saving to disk.

        Raises:
            RuntimeError: If called before the model has been initialized.
        """
        if self.model is None:
            raise RuntimeError("Cannot build artifact before model has been built.")

        z_min, z_max = self._compute_z_min_max(
            self.model,
            emb_matrix=self.emb_matrix,
            device=self.device,
        )

        return AutoencoderArtifact(
            state_dict={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            input_cols=tuple(self.input_cols),
            in_dim=self.in_dim,
            bottleneck_dim=self.cfg.bottleneck_dim,
            hidden_dim=self.cfg.hidden_dim,
            z_min=z_min,
            z_max=z_max,
        )

    def _save_best_checkpoint(self, save_dir: Path, save_name: str) -> AutoencoderArtifact:
        """
        Build and save an artifact from the current best model state.

        Args:
            save_dir: Directory in which to save the artifact file.
            save_name: Filename for the saved artifact.

        Returns:
            The constructed autoencoder artifact.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        artifact = self._build_best_artifact()
        artifact.save(save_dir / save_name)
        return artifact

    def train(
        self,
        *,
        save_best: bool = True,
        save_dir: str | Path = "checkpoints",
        save_name: str = "autoencoder_artifact.pt",
        verbose: bool = False,
    ) -> AutoencoderArtifact:
        """
        Run the full training loop and return the best saved artifact.

        Args:
            verbose: Whether to print training progress and display progress
                bars.

        Returns:
            The best-performing autoencoder artifact observed during training.

        Raises:
            RuntimeError: If model, optimizer, or dataloaders are not properly
                initialized, or if training finishes without producing a best
                artifact.
        """
        if self.seed is not None:
            self._set_seed(self.seed)

        self.train_loader, self.val_loader = self._build_loaders(seed=self.seed)
        self.model, self.optimizer = self._build_model_and_optimizer()

        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialized before training.")
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("Data loaders must be initialized before training.")

        best_val_loss = float("inf")
        best_state_dict: dict | None = None

        if verbose:
            print(
                f"Starting training for {self.cfg.num_epochs} epochs "
                f"on device={self.device} "
            )

        for epoch in range(self.cfg.num_epochs):
            train_metrics = self._train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.device,
                self.cfg.alpha,
                verbose=verbose,
            )
            val_metrics = self._evaluate_one_epoch(
                self.model,
                self.val_loader,
                self.device,
                self.cfg.alpha,
                verbose=verbose,
            )

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{self.cfg.num_epochs} | "
                    f"train_loss={train_metrics.loss:.4f} | "
                    f"train_recon={train_metrics.recon_loss:.4f} | "
                    f"train_cluster={train_metrics.cluster_loss:.4f} | "
                    f"train_acc={train_metrics.acc:.4f} | "
                    f"val_loss={val_metrics.loss:.4f} | "
                    f"val_recon={val_metrics.recon_loss:.4f} | "
                    f"val_cluster={val_metrics.cluster_loss:.4f} | "
                    f"val_acc={val_metrics.acc:.4f}"
                )

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if verbose:
                    print(f"New best validation loss: {best_val_loss:.4f}")

        if best_state_dict is None:
            raise RuntimeError("Training completed without producing a best artifact.")

        self.model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})

        if save_best:
            best_artifact = self._save_best_checkpoint(Path(save_dir), save_name)
        else:
            best_artifact = self._build_best_artifact()

        if verbose:
            print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

        return best_artifact