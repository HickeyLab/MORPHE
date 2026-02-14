from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from disco.core.latent_diffusion.data import Slice3DDataset
from disco.core.latent_diffusion.diffusion_trainer import DiffusionTrainer
from disco.core.latent_diffusion.strategy.base import DiffusionStrategy


@dataclass(frozen=True)
class ThreeDimensionalImputation(DiffusionStrategy):
    name: str = "three_dimensional_imputation"
    three_dimensional_cond_encoder: bool = True
    patience: int = 5
    decay_enabled: bool = False
    lr_decay_every: int = 10
    lr_decay_factor: float = 5
    
    def build_dataset(
        self, 
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:
        if not root_dir:
            raise ValueError("No root_dir provided.")
        return (Slice3DDataset(root_dir), Slice3DDataset(root_dir)) 
    

    def train_step(self, trainer: DiffusionTrainer, batch):
        """
        batch: (img_prev, img_next, img_mid)
        All tensors are on device (accelerator.prepare DataLoader handles pin).
        Behavior:
          - use prev as the base to add noise
          - use next as the condition (via cond_proj)
          - use mid as the target latent for loss
        """
        img_prev, img_next, img_mid, wp, wn = batch  # each: [B, 3, H, W]

        # encode latents (no grad)
        with torch.no_grad():
            latent_prev = self.vae.encode(img_prev).latent_dist.sample() * self.scaling_factor
            latent_next = self.vae.encode(img_next).latent_dist.sample() * self.scaling_factor
            latent_mid  = self.vae.encode(img_mid).latent_dist.sample() * self.scaling_factor

        # noise + timesteps
        noise = torch.randn_like(latent_prev)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latent_prev.shape[0],),
            device=latent_prev.device,
            dtype=torch.long
        )

        # add noise to the BASE latent (prev) — this is the noisy starting point
        noisy_latents = self.noise_scheduler.add_noise(latent_mid, noise, timesteps)

        # build condition tokens from next latent (use next as condition)
        wp = wp.view(-1, 1, 1, 1)   # [B,1,1,1]
        wn = wn.view(-1, 1, 1, 1)
        condition = self.cond_proj(wp*latent_prev + wn*latent_next)  # [B, num_tokens, cond_dim]

        # predict (model.sample follows previous pattern)
        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=condition).sample

        # MSE loss between predicted output and target (mid latent)
        # Mid as target, so compare to latent_mid
        loss = F.mse_loss(pred, latent_mid)
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
