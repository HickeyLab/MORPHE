from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from disco.core.latent_diffusion.data.builders import build_three_dim_dataset
from disco.core.latent_diffusion.train.base import LatentTrainStrategy

from disco.core.latent_diffusion.train.diffusion_trainer import DiffusionTrainer
from disco.utils import get_config_attr


@dataclass(frozen=True)
class ThreeDimImputationTrainStrategy(LatentTrainStrategy):
    three_dimensional_cond_encoder: bool = True
    
    supports_decay: bool = field(default=False, init=False)
    decay_enabled: bool = False
    
    patience: int | None = 5
    lr_decay_every: int | None = 10
    lr_decay_factor: float | None = 5
    
    def build_dataset(
        self, 
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:
        return build_three_dim_dataset(root_dir=root_dir)
    

    def train_step(
        self,
        trainer: "DiffusionTrainer",
        batch,
    ) -> torch.Tensor:
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

        # add noise to the BASE latent (prev) — this is the noisy starting point
        noisy_latents = trainer.noise_scheduler.add_noise(latent_mid, noise, timesteps) # type: ignore

        # build condition tokens from next latent (use next as condition)
        wp = wp.view(-1, 1, 1, 1)   # [B,1,1,1]
        wn = wn.view(-1, 1, 1, 1)
        condition = trainer.cond_proj(wp*latent_prev + wn*latent_next)  # [B, num_tokens, cond_dim]

        # predict (model.sample follows previous pattern)
        pred = trainer.unet(noisy_latents, timesteps, encoder_hidden_states=condition).sample

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
                loss = self.train_step(trainer, batch)
                tot += loss.item()
                cnt += 1
        return tot / max(cnt, 1)
