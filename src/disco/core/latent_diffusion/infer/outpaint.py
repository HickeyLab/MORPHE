from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from disco.constants import DIRECTION_TO_IDX
from disco.core.latent_diffusion.infer.base import BaseLatentInferencer
from disco.core.latent_diffusion.infer.run_config import OutpaintRunConfig
from disco.core.latent_diffusion.train.outpaint import OutpaintTrainStrategy
from src.disco.core.latent_diffusion.artifact import LatentDiffusionArtifact

from src.disco.viz.decoded_img import plot_decoded_image


class OutpaintInferencer(BaseLatentInferencer):
    def __init__(
        self, 
        *, 
        artifact: LatentDiffusionArtifact,
        train_strategy: OutpaintTrainStrategy, 
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(train_strategy, OutpaintTrainStrategy):
            raise TypeError(
                f"train_strategy must be an instance of OutpaintTrainStrategy, got {type(train_strategy)}"
            )
            
        super().__init__(
            artifact=artifact, 
            train_strategy=train_strategy, 
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )

        self.transform: torch.Callable[[Image.Image], torch.Tensor] = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def _extract_old_region(self, image_tensor: torch.Tensor, direction_idx: int, crop_ratio: float) -> torch.Tensor:
        """Apply zero mask to input image
        Args:
            image_tensor: Input tensor (1,3,H,W)
            direction: Mask direction (0-3)
            crop_ratio: Ratio of preserved area
        Returns:
            masked_tensor: Zero-masked image tensor
            mask_params: Parameters for latent mask calculation
        """
        b, c, h, w = image_tensor.shape
        masked = image_tensor.clone()

        if direction_idx in [0, 1]:  # Horizontal directions
            crop_w = int(w * crop_ratio)
            if direction_idx == 0:  # Preserve left
                masked = masked[..., :, :, :crop_w]
            else:  # Preserve right
                masked = masked[..., :, :, w - crop_w :]
        else:  # Vertical directions
            crop_h = int(h * crop_ratio)
            if direction_idx == 2:  # Preserve top
                masked = masked[..., :crop_h, :]
            else:  # Preserve bottom
                masked = masked[..., h - crop_h :, :]

        return masked

    def _create_latent_mask(self, bbox: torch.Tensor, latent_shape: Sequence[int]) -> torch.Tensor:
        b, _, lh, lw = latent_shape
        masks: list[torch.Tensor] = []

        for coords in bbox:
            x1 = coords[0] * lw
            y1 = coords[1] * lh
            x2 = coords[2] * lw
            y2 = coords[3] * lh

            xx, yy = torch.meshgrid(
                torch.arange(lw, device=self.device),
                torch.arange(lh, device=self.device),
            )
            mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
            masks.append(mask)

        return torch.stack(masks).unsqueeze(1)

    def _compute_mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return mean, std

    def _extract_new_region(self, generated: torch.Tensor, direction_idx: int, mask_size: int) -> torch.Tensor:
        if direction_idx == 0:  # right
            return generated[..., :, :, -mask_size:]
        if direction_idx == 1:  # left
            return generated[..., :, :, :mask_size]
        if direction_idx == 2:  # down
            return generated[..., :, -mask_size:, :]
        # direction_idx == 3: up
        return generated[..., :, :mask_size, :]

    def _stitch_image(self, combined: torch.Tensor, generated_patch: torch.Tensor, direction_idx: int) -> torch.Tensor:
        if direction_idx == 0:
            return torch.cat([combined, generated_patch], dim=-1)
        if direction_idx == 1:
            return torch.cat([generated_patch, combined], dim=-1)
        if direction_idx == 2:
            return torch.cat([combined, generated_patch], dim=-2)
        # direction_idx == 3
        return torch.cat([generated_patch, combined], dim=-2)

    def _cyclic_shift(self, generated: torch.Tensor, direction_idx: int, mask_size: int) -> torch.Tensor:
        if direction_idx == 0:  # right
            return torch.cat([generated[..., :, :, mask_size:], generated[..., :, :, :mask_size]], dim=-1)
        if direction_idx == 1:  # left
            return torch.cat([generated[..., :, :, -mask_size:], generated[..., :, :, :-mask_size]], dim=-1)
        if direction_idx == 2:  # down
            return torch.cat([generated[..., :, mask_size:, :], generated[..., :, :mask_size, :]], dim=-2)
        # direction_idx == 3: up
        return torch.cat([generated[..., :, -mask_size:, :], generated[..., :, :-mask_size, :]], dim=-2)

    def _cyclic_latent_shift(self, generated: torch.Tensor, direction_idx: int, mask_size: int) -> torch.Tensor:
        if direction_idx == 0:  # right
            return torch.cat([generated[..., :, :, mask_size:], generated[..., :, :, :mask_size]], dim=-1)
        if direction_idx == 1:  # left
            return torch.cat([generated[..., :, :, -mask_size:], generated[..., :, :, :-mask_size]], dim=-1)
        if direction_idx == 2:  # down
            return torch.cat([generated[..., :, mask_size:, :], generated[..., :, :mask_size, :]], dim=-2)
        # direction_idx == 3: up
        return torch.cat([generated[..., :, -mask_size:, :], generated[..., :, :-mask_size, :]], dim=-2)
    
    def _require_bbox_encoder(self) -> torch.nn.Module:
        if self.bbox_encoder is None:
            raise RuntimeError("bbox_encoder is required for outpaint inference.")
        return self.bbox_encoder

    def run_one(self, cfg: OutpaintRunConfig) -> torch.Tensor: 
        with Image.open(cfg.original_dir) as img:
            img = img.convert("RGB")
            image_tensor = self.transform(img).unsqueeze(0).to(self.device)

        direction_idx = DIRECTION_TO_IDX[cfg.direction]
        current_image = image_tensor.clone()
        b, c, h, w = current_image.shape
        
        # Fixed
        lh, lw = 64, 64

        crop_w, crop_h = int(w * cfg.crop_ratio), int(h * cfg.crop_ratio)
        crop_lw, crop_lh = int(lw * cfg.crop_ratio), int(lh * cfg.crop_ratio)
        mask_size = (w - crop_w) if cfg.direction in ["left", "right"] else (h - crop_h)
        latent_mask_size = (lw - crop_lw) if cfg.direction in ["left", "right"] else (lh - crop_lh)
        
        # Extract preserved region to initialize stitched image
        preserved_region = self._extract_old_region(current_image, direction_idx, cfg.crop_ratio)
        stitched = preserved_region
        current_latent: torch.Tensor | None = None

        # Step 1: Create bbox (normalized)
        if cfg.direction == "right":
            bbox = torch.tensor([[0.0, cfg.crop_ratio, 1.0, 1.0]], device=self.device)
        elif cfg.direction == "left":
            bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0 - cfg.crop_ratio]], device=self.device)
        elif cfg.direction == "down":
            bbox = torch.tensor([[cfg.crop_ratio, 0.0, 1.0, 1.0]], device=self.device)
        else:  # "up"
            bbox = torch.tensor([[0.0, 0.0, 1.0 - cfg.crop_ratio, 1.0]], device=self.device)

        run_dir = Path(cfg.save_dir) / cfg.save_name
        run_dir.mkdir(parents=True, exist_ok=True)

        last_generated_latent: torch.Tensor | None = None
        last_generated_img: torch.Tensor | None = None

        saved_latents: list[str] = []
        saved_images: list[str] = []

        for i in range(cfg.iterations):
            if i == 0:
                # Step 2: Create masked image
                mask = torch.zeros_like(current_image)
                x1 = int(bbox[0][0].item() * w)
                y1 = int(bbox[0][1].item() * h)
                x2 = int(bbox[0][2].item() * w)
                y2 = int(bbox[0][3].item() * h)
                mask[:, :, x1:x2, y1:y2] = 1
                masked_img = current_image * (1 - mask)
                
                # Step 3: Encode condition
                with torch.no_grad():
                    masked_latents = self.vae.encode(masked_img).latent_dist.sample()
                    masked_latents = masked_latents * self.scaling_factor
            else:
                masked_latents = current_latent
                
            # Step 4: Add noise and create latent_mask
            latent_mask = self._create_latent_mask(bbox, masked_latents.shape)

            noise = torch.randn_like(masked_latents)
            noisy_latents = self.noise_scheduler.add_noise(
                masked_latents * latent_mask,
                noise * latent_mask,
                torch.tensor(cfg.steps, device=self.device), # type: ignore
            )
            noisy_latents = masked_latents * (1 - latent_mask) + noisy_latents * latent_mask
            
            # Step 5: Denoising loop
            self.noise_scheduler.set_timesteps(cfg.steps)
            latent_input = noisy_latents
            
            bbox_encoder = self._require_bbox_encoder()
            condition = torch.cat(
                [
                    self.cond_proj(masked_latents),
                    bbox_encoder(bbox).unsqueeze(1).expand(-1, 64, -1),
                ],
                dim=-1,
            )

            for t in self.noise_scheduler.timesteps:
                latent_input = latent_input * latent_mask + masked_latents * (1 - latent_mask)
                with torch.no_grad():
                    noise_pred = self.unet(latent_input, t, encoder_hidden_states=condition).sample
                latent_input = self.noise_scheduler.step(noise_pred, t, latent_input).prev_sample # type: ignore
                
            # Step 6: Decode image
            with torch.no_grad():
                generated_latent = masked_latents * (1 - latent_mask) + latent_input * latent_mask
                generated_img = self.vae.decode(generated_latent / self.scaling_factor).sample

            last_generated_latent = generated_latent
            last_generated_img = generated_img

            # stitch
            new_patch = self._extract_new_region(generated_img, direction_idx, mask_size)
            stitched = self._stitch_image(stitched, new_patch, direction_idx)

            # save artifacts each iter
            latent_path = run_dir / f"{i:02d}_latent.pt"
            image_path = run_dir / f"{i:02d}_image.png"

            torch.save(generated_latent.detach().cpu(), latent_path)
            preview_uint8 = (
                ((generated_img[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255)
                .astype(np.uint8)
            )
            Image.fromarray(preview_uint8).save(image_path)

            saved_latents.append(str(latent_path))
            saved_images.append(str(image_path))

            current_image = self._cyclic_shift(generated_img, direction_idx, mask_size)
            current_latent = self._cyclic_latent_shift(generated_latent, direction_idx, latent_mask_size)
            
            # Display intermediate
            if cfg.show_plot:
                plot_decoded_image(
                    preview=preview_uint8,
                    iteration=i,
                    figsize=cfg.plot_fig_size,
                    title=cfg.plot_title,
                )

        stitched_preview_uint8 = (
            ((stitched[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        )

        return last_generated_latent if last_generated_latent is not None else (current_latent or torch.empty(0))

    def run(self, cfg: OutpaintRunConfig) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []

        files = sorted(p for p in cfg.original_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png") # type: ignore
        for fpath in files:
            per_file_save_name = str(Path(cfg.save_name) / fpath.stem)
            file_cfg = replace(
                cfg,
                original_dir=fpath,
                save_name=per_file_save_name,
            )
            res = self.run_one(file_cfg)
            results.append(res)

        return results
