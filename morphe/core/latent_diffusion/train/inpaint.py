from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ....constants import InferenceMode
from ..architecture import LatentArchitectureSpec
from ..data.builders import build_inpaint_dataset
from .base import LatentTrainTask
from .config import LatentTrainerConfig
from ....utils import get_config_attr

if TYPE_CHECKING:
    from .trainer import LatentDiffusionTrainer


@dataclass(frozen=True)
class InpaintTrainTask(LatentTrainTask):
    """
    Training strategy for latent diffusion-based image inpainting.

    This strategy defines:
    - Dataset construction for masked image inpainting
    - Architecture specification (UNet + encoders)
    - Training step logic for masked latent diffusion
    - Validation loop behavior

    Inpainting is performed by:
    - Encoding masked and target images into latent space
    - Applying diffusion noise only within masked regions
    - Conditioning on masked image latents and spatial coordinates
    """

    masks_per_image_train: int = 2
    masks_per_image_val: int = 5
    img_size: int = 512

    @property
    def inference_mode(self) -> InferenceMode:
        """Returns inference mode of this train strategy."""
        return InferenceMode.INPAINTING

    def build_architecture_spec(self, cfg: LatentTrainerConfig) -> LatentArchitectureSpec:
        """
        Construct the architecture specification used to build model components.

        This defines:
        - Pretrained model sources (UNet, VAE, scheduler)
        - Encoder types (conditioning + coordinate)
        - Encoder configuration kwargs

        Args:
            cfg: Trainer configuration containing architecture-related parameters.

        Returns:
            LatentArchitectureSpec used to instantiate all model components.
        """
        return LatentArchitectureSpec(
            unet_pretrained_path=cfg.unet_pretrained_path,
            ae_pretrained_path=cfg.ae_pretrained_path,
            scheduler_pretrained_path=cfg.scheduler_pretrained_path,
            cond_encoder_type="cond",
            coord_encoder_type="coord",
            bbox_encoder_type=None,
            cond_encoder_kwargs=cfg.cond_encoder_kwargs,
            coord_encoder_kwargs=cfg.coord_encoder_kwargs,
            bbox_encoder_kwargs=None,
        )

    def build_dataset(
        self,
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:
        """
        Build training and validation datasets for inpainting.

        Each sample consists of:
        - Masked input image
        - Ground-truth target image
        - Binary mask indicating inpaint regions

        Args:
            root_dir: Root directory containing dataset assets.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        return build_inpaint_dataset(
            root_dir=root_dir,
            masks_per_image_train=self.masks_per_image_train,
            masks_per_image_val=self.masks_per_image_val,
            img_size=self.img_size,
        )

    def train_step(
        self,
        trainer: "LatentDiffusionTrainer",
        batch,
    ) -> torch.Tensor:
        """
        Perform a single training step for inpainting.

        Pipeline:
        1. Encode target images into latent space
        2. Apply noise only within masked regions
        3. Encode conditioning inputs (masked image + coordinates)
        4. Predict noise with UNet
        5. Compute masked MSE loss

        Args:
            trainer: DiffusionTrainer containing models and runtime state.
            batch: Tuple of (masked_imgs, target_imgs, mask).

        Returns:
            Scalar loss tensor for the batch.

        Raises:
            ValueError: If required coordinate encoder is missing.
        """
        if not trainer.coord_encoder:
            raise ValueError("Coord encoder is required for inpainting training.")

        masked_imgs, target_imgs, mask = batch

        device = trainer.accelerator.device
        masked_imgs = masked_imgs.to(device)
        target_imgs = target_imgs.to(device)
        mask = mask.to(device)

        # Encode target latents
        with torch.no_grad():
            target_latents = trainer.encode_with_vae(target_imgs)

        B, C, lh, lw = target_latents.shape

        # Build latent mask
        latent_mask = F.interpolate(mask, size=(lh, lw), mode="nearest")
        latent_mask = latent_mask.expand(-1, C, -1, -1)

        # Diffusion forward
        noise = torch.randn_like(target_latents)

        num_train_timesteps = get_config_attr(
            trainer.noise_scheduler.config,
            "num_train_timesteps",
        )
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (B,),
            device=device,
        )

        noisy_latents = trainer.noise_scheduler.add_noise(
            target_latents * latent_mask,
            noise * latent_mask,
            timesteps,  # type: ignore
        )

        noisy_latents = target_latents * (1 - latent_mask) + noisy_latents

        # Condition encoding
        with torch.no_grad():
            masked_latents = trainer.encode_with_vae(masked_imgs)

        cond_tokens = trainer.cond_encoder(masked_latents)
        coord_tokens = trainer.coord_encoder(mask)

        condition = torch.cat([cond_tokens, coord_tokens], dim=-1)

        # UNet forward
        noise_pred = trainer.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=condition,
        ).sample

        # Loss
        loss = F.mse_loss(noise_pred * latent_mask, noise * latent_mask)

        return loss

    def validate_step(self, trainer: "LatentDiffusionTrainer") -> float:
        """
        Run full validation over the validation dataset.

        Reuses the training step (without gradients) to compute average loss.

        Args:
            trainer: DiffusionTrainer containing validation loader and models.

        Returns:
            Mean validation loss across all batches.
        """
        trainer.unet.eval()

        total = 0.0
        count = 0

        with torch.no_grad():
            for batch in trainer.val_loader:
                loss = self.train_step(trainer, batch)
                total += loss.item()
                count += 1

        return total / max(count, 1)