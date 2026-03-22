from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from disco.config import InferenceMode
from disco.core.latent_diffusion.architecture import LatentArchitectureSpec
from disco.core.latent_diffusion.data.builders import build_three_dim_dataset
from disco.core.latent_diffusion.train.base import LatentTrainStrategy

from disco.core.latent_diffusion.train.diffusion_trainer import DiffusionTrainer
from disco.core.latent_diffusion.train.train_config import LatentTrainerConfig
from disco.utils import get_config_attr


@dataclass(frozen=True)
class ThreeDimImputationTrainStrategy(LatentTrainStrategy):
    """
    Training strategy for 3D-aware latent diffusion imputation.

    This strategy learns to reconstruct a missing/intermediate frame
    (mid) given temporal or spatial context from neighboring frames
    (prev and next).

    Key characteristics:
    - Uses weighted combination of neighboring latents as conditioning
    - Predicts the target latent directly (not noise)
    - Supports learning rate decay (supports_decay = True)

    Typical use case:
    - Temporal interpolation
    - Slice imputation in volumetric / 3D data
    """
    supports_decay: bool = True
    
    @property  
    def inference_mode(self) -> InferenceMode:
        """Returns inference mode of this train strategy."""
        return InferenceMode.THREE_DIMENSIONAL_IMPUTATION
    
    def build_architecture_spec(self, cfg: LatentTrainerConfig,) -> LatentArchitectureSpec:
        """
        Construct the architecture specification for 3D imputation.

        This configuration:
        - Uses a 3D conditioning encoder (`cond3d`)
        - Does not require coordinate or bounding-box encoders
        - Uses standard latent diffusion components (UNet, VAE, scheduler)

        Args:
            cfg: Trainer configuration containing architecture parameters.

        Returns:
            LatentArchitectureSpec defining all model components.
        """
        return LatentArchitectureSpec(
            unet_pretrained_path=cfg.unet_pretrained_path,
            ae_pretrained_path=cfg.ae_pretrained_path,
            scheduler_pretrained_path=cfg.scheduler_pretrained_path,
            cond_encoder_type="cond3d",
            coord_encoder_type=None,
            bbox_encoder_type=None,
            cond_encoder_kwargs=cfg.cond_encoder_kwargs,
            coord_encoder_kwargs=None,
            bbox_encoder_kwargs=None,
        )
    
    def build_dataset(
        self, 
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:
        """
        Build training and validation datasets for 3D imputation.

        Each sample consists of:
        - Previous frame/image
        - Next frame/image
        - Target intermediate frame/image
        - Weight coefficients for combining context (wp, wn)

        Args:
            root_dir: Root directory containing dataset assets.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        return build_three_dim_dataset(root_dir=root_dir)
    
    def train_step(
        self,
        trainer: "DiffusionTrainer",
        batch,
    ) -> torch.Tensor:
        """
        Perform a single training step for 3D imputation.

        Pipeline:
        1. Encode previous, next, and target images into latent space
        2. Add noise to the target (mid) latent
        3. Construct conditioning from weighted combination of prev/next latents
        4. Predict target latent using UNet
        5. Compute MSE loss against ground-truth latent

        Args:
            trainer: DiffusionTrainer containing models and runtime state.
            batch: Tuple of:
                - img_prev: Previous image tensor (B, C, H, W)
                - img_next: Next image tensor (B, C, H, W)
                - img_mid: Target image tensor (B, C, H, W)
                - wp: Weight for previous latent (B,)
                - wn: Weight for next latent (B,)

        Returns:
            Scalar loss tensor for the batch.

        Notes:
            - Conditioning is computed as: wp * prev + wn * next
            - Model predicts the clean latent (not noise)
        """
        img_prev, img_next, img_mid, wp, wn = batch  # each: [B, 3, H, W]

        # encode latents (no grad)
        with torch.no_grad():
            latent_prev = trainer.vae.encode(img_prev).latent_dist.sample() * trainer.scaling_factor # type: ignore
            latent_next = trainer.vae.encode(img_next).latent_dist.sample() * trainer.scaling_factor # type: ignore
            latent_mid  = trainer.vae.encode(img_mid).latent_dist.sample() * trainer.scaling_factor # type: ignore

        # noise + timesteps
        noise = torch.randn_like(latent_prev)
        num_train_timesteps = get_config_attr(trainer.noise_scheduler.config, "num_train_timesteps")
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (latent_prev.shape[0],),
            device=latent_prev.device,
            dtype=torch.long
        )

        noisy_latents = trainer.noise_scheduler.add_noise(latent_mid, noise, timesteps) # type: ignore

        wp = wp.view(-1, 1, 1, 1)
        wn = wn.view(-1, 1, 1, 1)
        condition = trainer.cond_encoder(wp * latent_prev + wn * latent_next)

        pred = trainer.unet(noisy_latents, timesteps, encoder_hidden_states=condition).sample

        loss = F.mse_loss(pred, latent_mid)
        return loss

    def validate_step(self, trainer: DiffusionTrainer) -> float:
        """
        Run validation over the entire validation dataset.

        Reuses the training step (without gradients) to compute
        average reconstruction loss.

        Args:
            trainer: DiffusionTrainer containing validation loader and models.

        Returns:
            Mean validation loss across all batches.
        """
        trainer.unet.eval()
        tot = 0
        cnt = 0
        with torch.no_grad():
            for batch in trainer.val_loader:
                loss = self.train_step(trainer, batch)
                tot += loss.item()
                cnt += 1
        return tot / max(cnt, 1)