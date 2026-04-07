from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm

from ....constants import InferenceMode
from ..architecture import LatentArchitectureSpec
from ..data.builders import build_outpaint_dataset
from .base import LatentTrainTask
from .train_config import LatentTrainerConfig
from ....utils import get_config_attr
if TYPE_CHECKING:
    from .trainer import LatentDiffusionTrainer



@dataclass(frozen=True)
class OutpaintTrainTask(LatentTrainTask):
    """
    Training strategy for latent diffusion-based image outpainting.

    This strategy extends images beyond their original boundaries by:
    - Masking out regions to be generated (outside context)
    - Conditioning on the visible image content
    - Using bounding box embeddings to specify outpaint regions

    Key components:
    - Conditioning encoder (masked image latents)
    - Coordinate encoder (optional spatial context)
    - Bounding-box encoder (region specification)
    """

    masks_per_image_train: int = 5
    masks_per_image_val: int = 5
    img_size: int = 512
    
    @property
    def inference_mode(self) -> InferenceMode:
        """Returns inference mode of this train strategy."""
        return InferenceMode.OUTPAINTING
    
    def build_architecture_spec(self, cfg: LatentTrainerConfig,) -> LatentArchitectureSpec:
        """
        Construct the architecture specification for outpainting.

        This enables:
        - Bounding-box conditioning
        - Spatial awareness via coordinate encoding
        - Standard latent diffusion components (UNet, VAE, scheduler)

        Args:
            cfg: Trainer configuration containing architecture parameters.

        Returns:
            LatentArchitectureSpec defining all model components.
        """
        return LatentArchitectureSpec(
            unet_pretrained_path=cfg.unet_pretrained_path,
            ae_pretrained_path=cfg.ae_pretrained_path,
            scheduler_pretrained_path=cfg.scheduler_pretrained_path,
            cond_encoder_type="cond",
            coord_encoder_type="coord",
            bbox_encoder_type="bbox",
            cond_encoder_kwargs=cfg.cond_encoder_kwargs,
            coord_encoder_kwargs=cfg.coord_encoder_kwargs,
            bbox_encoder_kwargs=cfg.bbox_encoder_kwargs,
        )

    def build_dataset(
        self,
        root_dir: Path,
    ) -> tuple[Dataset, Dataset]:
        """
        Build training and validation datasets for outpainting.

        Each sample consists of:
        - Masked input image (context region)
        - Ground-truth full image
        - Bounding box specifying region to generate

        Args:
            root_dir: Root directory containing dataset assets.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        return build_outpaint_dataset(
            root_dir=root_dir,
            masks_per_image_train=self.masks_per_image_train,
            masks_per_image_val=self.masks_per_image_val
        )

    def _create_latent_mask(self, bbox, latent_shape, device):
        """
        Create a binary latent-space mask from bounding box coordinates.

        The mask defines regions where diffusion noise is applied (i.e.,
        regions to be generated during outpainting).

        Args:
            bbox: Tensor of shape (B, 4) with normalized coordinates
                  (x1, y1, x2, y2) in [0, 1].
            latent_shape: Shape of latent tensor (B, C, H, W).
            device: Device on which to create the mask.

        Returns:
            Tensor of shape (B, 1, H, W) with values:
            - 1.0 in outpaint regions
            - 0.0 elsewhere
        """
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

    def train_step(self, trainer: "LatentDiffusionTrainer", batch):
        """
        Perform a single training step for outpainting.

        Pipeline:
        1. Encode target image into latent space
        2. Construct latent mask from bounding boxes
        3. Apply noise only within masked (outpaint) regions
        4. Encode conditioning inputs (masked image + bbox)
        5. Predict noise with UNet
        6. Compute masked MSE loss

        Args:
            trainer: DiffusionTrainer containing models and runtime state.
            batch: Tuple of (masked_img, target_img, bbox).

        Returns:
            Scalar loss tensor for the batch.

        Raises:
            ValueError: If bounding-box encoder is not available.
        """
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
                trainer.cond_encoder(masked_lat),
                cond_bbox.unsqueeze(1).expand(-1, 64, -1),
            ],
            dim=-1,
        )

        pred = trainer.unet(noisy, t, encoder_hidden_states=cond_tokens).sample

        loss = F.mse_loss(pred * mask, noise * mask)

        return loss

    def validate_step(self, trainer: "LatentDiffusionTrainer") -> float:
        """
        Run validation over the entire validation dataset.

        Reuses the training step (without gradients) to compute the
        average masked diffusion loss.

        Args:
            trainer: DiffusionTrainer containing validation loader and models.

        Returns:
            Mean validation loss across all batches.
        """
        trainer.unet.eval()

        total = 0.0
        count = 0

        with torch.no_grad():
            for batch in tqdm(trainer.val_loader):
                loss = self.train_step(trainer, batch)
                total += loss.item()
                count += 1

        return total / max(count, 1)