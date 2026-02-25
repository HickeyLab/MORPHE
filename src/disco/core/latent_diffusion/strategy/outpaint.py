from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm

from src.disco.core.latent_diffusion.data import OutpaintDataset
from src.disco.core.latent_diffusion.diffusion_trainer import DiffusionTrainer
from src.disco.core.latent_diffusion.strategy.base import DiffusionStrategy


@dataclass(frozen=True)
class OutpaintDiffusion(DiffusionStrategy):
    name: str = "outpaint_diffusion"
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
    
    def _create_latent_mask(bbox, latent_shape, device):
        b, _, H, W = latent_shape
        masks = []

        for coords in bbox:
            x1, y1, x2, y2 = coords * torch.tensor([W, H, W, H], device=device)
            xx, yy = torch.meshgrid(
                torch.arange(W, device=device),
                torch.arange(H, device=device)
            )
            mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
            masks.append(mask)

        return torch.stack(masks).unsqueeze(1)

    def train_step(self, trainer: DiffusionTrainer, batch):
        masked_img, target_img, bbox = batch

        with torch.no_grad():
            target_lat = trainer.vae.encode(target_img).latent_dist.sample()
            target_lat = target_lat * trainer.vae.config.scaling_factor

        mask = self._create_latent_mask(bbox, target_lat.shape, target_lat.device)

        noise = torch.randn_like(target_lat)
        t = torch.randint(0, trainer.noise_scheduler.config.num_train_timesteps,
                          (target_lat.size(0),), device=target_lat.device)

        noisy = trainer.noise_scheduler.add_noise(target_lat * mask, noise * mask, t)
        noisy = target_lat * (1 - mask) + noisy

        with torch.no_grad():
            masked_lat = trainer.vae.encode(masked_img).latent_dist.sample()
            masked_lat = masked_lat * trainer.vae.config.scaling_factor

        cond_bbox = trainer.coord_encoder(bbox)
        cond_tokens = torch.cat([
            trainer.cond_proj(masked_lat),
            cond_bbox.unsqueeze(1).expand(-1, 64, -1)
        ], dim=-1)

        pred = trainer.unet(noisy, t, encoder_hidden_states=cond_tokens).sample

        loss = F.mse_loss(pred * mask, noise * mask)
        return loss

    def validate_step(self, trainer: DiffusionTrainer) -> float:
        trainer.unet.eval()
        total = 0
        with torch.no_grad():
            for batch in tqdm(trainer.val_loader):
                total += self.train_step(batch).item()
        return total / len(trainer.val_loader)
