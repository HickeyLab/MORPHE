from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from accelerate import Accelerator
from tqdm import tqdm

from src.disco.core.latent_diffusion.artifact import LatentDiffuserArtifact
from src.disco.core.latent_diffusion.model import (
    CondEncoder,
    CoordEncoder,
    BBoxEncoder,
    CondEncoder3D,
)
from src.disco.core.latent_diffusion.strategy.base import DiffusionStrategy
from src.disco.viz.loss_curve import plot_loss_curve


def _get_scaling_factor(vae: AutoencoderKL) -> float:
    return float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))


class DiffusionTrainer:

    def __init__(
        self,
        *,
        strategy: DiffusionStrategy,
        root_dir: Path,
        save_dir: str = "checkpoints",
        save_best_only: bool = True,
        pretrained: str = "runwayml/stable-diffusion-v1-5",
        lr: float = 2e-5,
        mixed_precision: str = "fp16",
        grad_clip: float = 1.0,
        cond_encoder_kwargs: dict | None = None,
        coord_encoder_kwargs: dict | None = None,
        bbox_encoder_kwargs: dict | None = None,
        batch_size: int = 8,
        val_batch_size: int = 8,
    ):

        self.strategy = strategy
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.grad_clip = float(grad_clip)

        self.cond_encoder_kwargs = cond_encoder_kwargs or {}
        self.coord_encoder_kwargs = coord_encoder_kwargs or {}
        self.bbox_encoder_kwargs = bbox_encoder_kwargs or {}

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self.accelerator = Accelerator(mixed_precision=mixed_precision)

        # -------------------------------
        # Dataset
        # -------------------------------
        train_data, val_data = strategy.build_dataset(root_dir)

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=strategy.train_num_workers,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_data,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=strategy.val_num_workers,
            pin_memory=True,
        )

        # -------------------------------
        # Load pretrained backbone
        # -------------------------------
        self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained, subfolder="scheduler"
        )

        if strategy.name == "3dimputation":
            self.noise_scheduler.config.prediction_type = "sample"
        if not strategy.requires_bbox_encoder and not strategy.requires_bbox_encoder:
            if hasattr(self.noise_scheduler, "config"):
                self.noise_scheduler.config.prediction_type = "sample"

        # -------------------------------
        # Condition encoders
        # -------------------------------
        if strategy.requires_bbox_encoder:
            self.bbox_encoder = BBoxEncoder(**self.bbox_encoder_kwargs)
            self.cond_proj = CondEncoder(
                in_channels=4,
                out_channels=736,
                **self.cond_encoder_kwargs,
            )
            self.coord_encoder = None

        elif strategy.requires_coord_encoder:
            self.coord_encoder = CoordEncoder(**self.coord_encoder_kwargs)
            self.cond_proj = CondEncoder(
                in_channels=4,
                out_channels=736,
                **self.cond_encoder_kwargs,
            )
            self.bbox_encoder = None
        else:
            self.coord_encoder = None
            self.bbox_encoder = None
            self.cond_proj = CondEncoder3D(
                in_channels=4,
                out_channels=768,
                **self.cond_encoder_kwargs,
            )

        # Freeze VAE
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling_factor = _get_scaling_factor(self.vae)

        # -------------------------------
        # Optimizer
        # -------------------------------
        params = list(self.unet.parameters()) + list(self.cond_proj.parameters())
        if self.coord_encoder is not None:
            params += list(self.coord_encoder.parameters())
        if self.bbox_encoder is not None:
            params += list(self.bbox_encoder.parameters())

        self.optimizer = torch.optim.AdamW(params, lr=lr)

        # -------------------------------
        # Prepare with accelerator
        # -------------------------------
        if self.bbox_encoder is not None:
            (
                self.vae,
                self.unet,
                self.cond_proj,
                self.bbox_encoder,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            ) = self.accelerator.prepare(
                self.vae,
                self.unet,
                self.cond_proj,
                self.bbox_encoder,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            )

        elif self.coord_encoder is not None:
            (
                self.vae,
                self.unet,
                self.cond_proj,
                self.coord_encoder,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            ) = self.accelerator.prepare(
                self.vae,
                self.unet,
                self.cond_proj,
                self.coord_encoder,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            )
        else:
            (
                self.vae,
                self.unet,
                self.cond_proj,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            ) = self.accelerator.prepare(
                self.vae,
                self.unet,
                self.cond_proj,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            )

        os.makedirs(self.save_dir, exist_ok=True)

    # =====================================================
    # Training
    # =====================================================

    def _train_one_epoch(self, *, epoch: int) -> float:

        self.unet.train()
        losses: list[float] = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:

            with self.accelerator.accumulate(self.unet):

                loss = self.strategy.train_step(self, batch)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.grad_clip is not None:
                    params = self.optimizer.param_groups[0]["params"]
                    self.accelerator.clip_grad_norm_(params, self.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            lv = float(loss.detach().mean().item())
            losses.append(lv)
            pbar.set_postfix({"loss": float(np.mean(losses))})

        return float(np.mean(losses)) if losses else float("nan")

    # =====================================================
    # Save
    # =====================================================

    def _save_best_checkpoint(self, save_dir: Path):

        save_dir.mkdir(exist_ok=True)

        coord_state = None
        if self.coord_encoder is not None:
            coord_state = self.coord_encoder.state_dict()

        bbox_state = None
        if self.bbox_encoder is not None:
            bbox_state = self.bbox_encoder.state_dict()

        # FIX: FrozenDict compatibility
        cfg = self.unet.config
        if hasattr(cfg, "to_dict"):
            cfg = cfg.to_dict()
        else:
            cfg = dict(cfg)

        artifact = LatentDiffuserArtifact(
            unet_state_dict=self.unet.state_dict(),
            cond_encoder_state_dict=self.cond_proj.state_dict(),
            coord_encoder_state_dict=coord_state,
            bbox_encoder_state_dict=bbox_state,
            unet_config=cfg,
            cond_encoder_kwargs=dict(self.cond_encoder_kwargs),
            coord_encoder_kwargs=(
                None if self.coord_encoder is None else dict(self.coord_encoder_kwargs)
            ),
            bbox_encoder_kwargs=(
                None if self.bbox_encoder is None else dict(self.bbox_encoder_kwargs)
            ),
        )

        artifact_path = save_dir / "latent_diffuser_artifact.pt"
        artifact.save(artifact_path)

    # =====================================================
    # Main Loop
    # =====================================================

    def train(
        self,
        *,
        epochs: int = 20,
        show_loss_curve: bool = False,
        figsize: tuple[int, int] = (7, 5),
        save_checkpoints: bool = True,
    ) -> None:

        best_val = float("inf")
        patience_cnt = 0
        patience = getattr(self.strategy, "patience", None)

        for epoch in range(epochs):

            train_loss = self._train_one_epoch(epoch=epoch)
            self.train_loss_history.append(train_loss)

            val_loss = self.strategy.validate_step(trainer=self)
            self.val_loss_history.append(val_loss)

            print(
                f"Epoch {epoch} -> "
                f"Train: {train_loss:.6f}  Val: {val_loss:.6f}"
            )

            improved = val_loss < best_val

            if improved:
                best_val = val_loss
                self.accelerator.wait_for_everyone()

                if save_checkpoints:
                    self._save_best_checkpoint(self.save_dir)

            if patience is not None:
                if improved:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        print("Early stopping.")
                        break

            if getattr(self.strategy, "decay_enabled", False):
                every = getattr(self.strategy, "lr_decay_every", None)
                factor = float(getattr(self.strategy, "lr_decay_factor", 0.5))
                if every is not None and epoch % every == 0:
                    for g in self.optimizer.param_groups:
                        g["lr"] *= factor

            if show_loss_curve:
                plot_loss_curve(
                    train_losses=self.train_loss_history,
                    val_losses=self.val_loss_history,
                    figsize=figsize,
                    title="Loss Curve",
                    save_path=self.save_dir,
                )