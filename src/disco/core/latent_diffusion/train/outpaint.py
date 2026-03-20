from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm

from disco.core.latent_diffusion.data.builders import build_outpaint_dataset
from disco.core.latent_diffusion.train.base import LatentTrainStrategy
from disco.utils import get_config_attr

if TYPE_CHECKING:
    from disco.core.latent_diffusion.train.diffusion_trainer import DiffusionTrainer


@dataclass(frozen=True)
class OutpaintTrainStrategy(LatentTrainStrategy):
    train_num_workers: int = 4
    val_num_workers: int = 2

    # task-specific
    masks_per_image_train: int = 5
    masks_per_image_val: int = 5
    img_size: int = 512

    requires_coord_encoder: bool = False
    requires_bbox_encoder: bool = True
    three_dimensional_cond_encoder: bool = False

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    def build_dataset(
        self,
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:
        return build_outpaint_dataset(
            root_dir=root_dir,
            masks_per_image_train=self.masks_per_image_train,
            masks_per_image_val=self.masks_per_image_val
        )

    # --------------------------------------------------
    # Create latent mask
    # --------------------------------------------------
    def _create_latent_mask(self, bbox, latent_shape, device):

        _, _, H, W = latent_shape

        masks = []

        for coords in bbox:
            x1 = int(coords[0] * W)
            y1 = int(coords[1] * H)
            x2 = int(coords[2] * W)
            y2 = int(coords[3] * H)

            mask = torch.zeros((H, W), device=device)
            mask[y1:y2, x1:x2] = 1.0
            masks.append(mask)

        mask = torch.stack(masks).unsqueeze(1)  # (B,1,H,W)
        return mask

    # --------------------------------------------------
    # Train step
    # --------------------------------------------------
    def train_step(self, trainer: "DiffusionTrainer", batch):
        if not trainer.bbox_encoder:
            raise ValueError("BBox encoder is required for outpainting training.")
        
        masked_img, target_img, bbox = batch

        device = trainer.accelerator.device
        masked_img = masked_img.to(device)
        target_img = target_img.to(device)
        bbox = bbox.to(device)

        # encode target
        with torch.no_grad():
            target_lat = trainer.vae.encode(target_img).latent_dist.sample() # type: ignore
            target_lat = target_lat * trainer.scaling_factor

        # create mask
        mask = self._create_latent_mask(bbox, target_lat.shape, device)

        noise = torch.randn_like(target_lat)
        
        num_train_timesteps = get_config_attr(trainer.noise_scheduler.config, "num_train_timesteps")
        t = torch.randint(
            0,
            num_train_timesteps,
            (target_lat.size(0),),
            device=device,
        )

        noisy = trainer.noise_scheduler.add_noise(
            target_lat * mask,
            noise * mask,
            t, # type: ignore
        )

        noisy = target_lat * (1 - mask) + noisy

        # encode masked image
        with torch.no_grad():
            masked_lat = trainer.vae.encode(masked_img).latent_dist.sample() # type: ignore
            masked_lat = masked_lat * trainer.scaling_factor

        cond_bbox = trainer.bbox_encoder(bbox)

        cond_tokens = torch.cat(
            [
                trainer.cond_proj(masked_lat),
                cond_bbox.unsqueeze(1).expand(-1, 64, -1),
            ],
            dim=-1,
        )

        pred = trainer.unet(noisy, t, encoder_hidden_states=cond_tokens).sample

        loss = F.mse_loss(pred * mask, noise * mask)

        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validate_step(self, trainer: "DiffusionTrainer") -> float:

        trainer.unet.eval()

        total = 0.0
        count = 0

        with torch.no_grad():
            for batch in tqdm(trainer.val_loader):
                loss = self.train_step(trainer, batch)
                total += loss.item()
                count += 1

        return total / max(count, 1)