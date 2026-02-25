from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.disco.core.latent_diffusion.data import InpaintDataset
from src.disco.core.latent_diffusion.strategy.base import DiffusionStrategy

if TYPE_CHECKING:
    from src.disco.core.latent_diffusion.diffusion_trainer import DiffusionTrainer


@dataclass(frozen=True)
class ArbitraryInpainting(DiffusionStrategy):

    # --------------------------------------------------
    # Metadata (kept)
    # --------------------------------------------------
    name: str = "arbitrary_inpainting"
    requires_coord_encoder: bool = True
    three_dimensional_cond_encoder: bool = False

    # workers (kept)
    train_num_workers: int = 4
    val_num_workers: int = 2

    # training control (kept)
    patience: Optional[int] = None
    decay_enabled: Optional[bool] = None
    lr_decay_every: Optional[int] = None
    lr_decay_factor: Optional[float] = None

    # task-specific params
    masks_per_image_train: int = 2
    masks_per_image_val: int = 5
    img_size: int = 512

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    def build_dataset(
        self,
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:

        train_dir = root_dir / "train"
        val_dir = root_dir / "val"

        train_dataset = InpaintDataset(
            train_dir,
            img_size=self.img_size,
            masks_per_image=self.masks_per_image_train,
        )

        val_dataset = InpaintDataset(
            val_dir,
            img_size=self.img_size,
            masks_per_image=self.masks_per_image_val,
        )

        return train_dataset, val_dataset

    # --------------------------------------------------
    # FULL train step
    # --------------------------------------------------
    def train_step(
        self,
        trainer: "DiffusionTrainer",
        batch,
    ) -> torch.Tensor:

        masked_imgs, target_imgs, mask = batch

        device = trainer.accelerator.device
        masked_imgs = masked_imgs.to(device)
        target_imgs = target_imgs.to(device)
        mask = mask.to(device)

        # ---------------------------------------
        # Encode target latents
        # ---------------------------------------
        with torch.no_grad():
            target_latents = trainer.vae.encode(target_imgs).latent_dist.sample()
            target_latents = target_latents * trainer.scaling_factor

        B, C, lh, lw = target_latents.shape

        # ---------------------------------------
        # Build latent mask
        # ---------------------------------------
        latent_mask = F.interpolate(mask, size=(lh, lw), mode="nearest")
        latent_mask = latent_mask.expand(-1, C, -1, -1)

        # ---------------------------------------
        # Diffusion forward
        # ---------------------------------------
        noise = torch.randn_like(target_latents)

        timesteps = torch.randint(
            0,
            trainer.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        )

        noisy_latents = trainer.noise_scheduler.add_noise(
            target_latents * latent_mask,
            noise * latent_mask,
            timesteps,
        )

        noisy_latents = target_latents * (1 - latent_mask) + noisy_latents

        # ---------------------------------------
        # Condition encoding
        # ---------------------------------------
        with torch.no_grad():
            masked_latents = trainer.vae.encode(masked_imgs).latent_dist.sample()
            masked_latents = masked_latents * trainer.scaling_factor

        cond_tokens = trainer.cond_proj(masked_latents)
        coord_tokens = trainer.coord_encoder(mask)

        condition = torch.cat([cond_tokens, coord_tokens], dim=-1)

        # ---------------------------------------
        # UNet forward
        # ---------------------------------------
        noise_pred = trainer.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=condition,
        ).sample

        # ---------------------------------------
        # Loss
        # ---------------------------------------
        loss = F.mse_loss(noise_pred * latent_mask, noise * latent_mask)

        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validate_step(self, trainer: "DiffusionTrainer") -> float:

        trainer.unet.eval()

        total = 0.0
        count = 0

        with torch.no_grad():
            for batch in trainer.val_loader:
                loss = self.train_step(trainer, batch)
                total += loss.item()
                count += 1

        return total / max(count, 1)