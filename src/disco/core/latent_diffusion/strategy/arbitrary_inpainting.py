from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from disco.core.latent_diffusion.data import OutpaintDataset
from disco.core.latent_diffusion.diffusion_trainer import DiffusionTrainer
from disco.core.latent_diffusion.strategy.base import DiffusionStrategy


@dataclass(frozen=True)
class ArbitraryInpainting(DiffusionStrategy):
    name: str = "arbitrary_inpainting"
    masks_per_image_train: int = 50,
    masks_per_image_val: int = 5,
    img_size: int = 512,
    
    def build_dataset(
        self, 
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:
        if not root_dir:
            raise ValueError("No root_dir provided.")
        return (OutpaintDataset(root_dir, self.img_size, self.masks_per_image_train), OutpaintDataset(root_dir, self.img_size, self.masks_per_image_val)) 
    

    def train_step(self, trainer: DiffusionTrainer, batch):
        masked_imgs, target_imgs, mask = batch

        # target latents
        with torch.no_grad():
            target_latents = trainer.vae.encode(target_imgs).latent_dist.sample()
            target_latents = target_latents * trainer.vae.config.scaling_factor

        B, C, lh, lw = target_latents.shape

        latent_mask = F.interpolate(mask, size=(lh, lw), mode="nearest")
        latent_mask = latent_mask.expand(-1, target_latents.shape[1], -1, -1)  # (B,4,lh,lw)

        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0, trainer.noise_scheduler.config.num_train_timesteps,
            (B,), device=target_latents.device
        )

        noisy_latents = trainer.noise_scheduler.add_noise(
            target_latents * latent_mask,
            noise * latent_mask,
            timesteps
        )
        noisy_latents = target_latents * (1-latent_mask) + noisy_latents

        # masked latents for CondEncoder
        with torch.no_grad():
            masked_latents = trainer.vae.encode(masked_imgs).latent_dist.sample()
            masked_latents = masked_latents * trainer.vae.config.scaling_factor

        # Encode conditions
        cond_tokens = trainer.cond_proj(masked_latents)      # (B,64,736)
        coord_tokens = trainer.coord_encoder(mask)         # (B,64,32)

        condition = torch.cat([cond_tokens, coord_tokens], dim=-1)  # (B,64,768)

        noise_pred = trainer.unet(noisy_latents, timesteps, encoder_hidden_states=condition).sample

        loss = F.mse_loss(noise_pred * latent_mask, noise * latent_mask)
        return loss

    def validate_step(self, trainer: DiffusionTrainer) -> float:
        trainer.unet.eval()
        tot = 0
        cnt = 0
        with torch.no_grad():
            for batch in trainer.val_loader:
                loss = self.train_step(batch)
                tot += loss.item()
                cnt += 1
        return tot / max(cnt, 1)