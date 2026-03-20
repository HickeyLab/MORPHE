from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from disco.core.latent_diffusion.infer.base import (
    BaseLatentInferencer
)
from disco.core.latent_diffusion.infer.run_config import InpaintRunConfig
from disco.core.latent_diffusion.model import CoordEncoder
from disco.core.latent_diffusion.train.inpaint import InpaintTrainStrategy
from src.disco.core.latent_diffusion.artifact import LatentDiffusionArtifact
from src.disco.viz.decoded_img import plot_decoded_image, plot_inpainting_triplet


class InpaintInferencer(BaseLatentInferencer):
    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        train_strategy: InpaintTrainStrategy,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(train_strategy, InpaintTrainStrategy):
            raise TypeError(
                f"train_strategy must be an instance of InpaintTrainStrategy, got {type(train_strategy)}"
            )

        super().__init__(
            artifact=artifact,
            train_strategy=train_strategy,
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )

        if self.coord_encoder is None:
            raise RuntimeError(
                "InpaintInferencer requires coord_encoder and bbox_encoder to be present in the artifact."
            )

    def _load_image(self, path: str | Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.train_strategy.img_size, self.train_strategy.img_size))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # (3,H,W)
        img = img * 2 - 1  # [-1, 1]
        return img.unsqueeze(0).to(self.device)  # (1,3,512,512)

    def _load_mask(self, path: str | Path) -> torch.Tensor:
        """
        mask: white=mask=1, black=keep=0
        """
        m = (
            Image.open(path)
            .convert("L")
            .resize((self.train_strategy.img_size, self.train_strategy.img_size))
        )
        m = np.array(m).astype(np.float32) / 255.0
        m = torch.tensor(m).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return m.to(self.device)

    def _require_coord_encoder(self) -> CoordEncoder:
        if self.coord_encoder is None:
            raise RuntimeError("coord_encoder is required for inpaint inference.")
        return self.coord_encoder

    def _to_numpy_image(self, x: torch.Tensor) -> np.ndarray:
        x = x.detach().cpu()

        if x.ndim == 4:  # (B, C, H, W)
            x = x[0]

        if x.ndim == 3 and x.shape[0] in {1, 3}:  # (C, H, W) → (H, W, C)
            x = x.permute(1, 2, 0)

        return x.numpy()

    def run_one(self, cfg: InpaintRunConfig) -> "torch.Tensor":
        image = self._load_image(cfg.image_dir)
        mask = self._load_mask(cfg.mask_dir)

        # 1. Encode image to latent
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample() * self.scaling_factor  # type: ignore

        B, C, H, W = latent.shape

        # Expand mask to 4 latent channels
        latent_mask = F.interpolate(mask, size=(H, W), mode="nearest")
        latent_mask = latent_mask.expand(-1, C, -1, -1)

        # mask the latent
        masked_latent = latent * (1 - latent_mask)

        # show masked decode
        if cfg.show_plot:
            with torch.no_grad():
                m = self.vae.decode(masked_latent / self.scaling_factor).sample  # type: ignore
            m = (m.clamp(-1, 1) + 1) / 2
            m = m[0].permute(1, 2, 0).detach().cpu().numpy()
            plot_decoded_image(
                preview=m,
                figsize=cfg.plot_fig_size,
                title=cfg.plot_title,
            )

        # DDPM scheduler
        self.noise_scheduler.set_timesteps(cfg.num_steps, device=self.device)

        # Initial noise
        noisy = torch.randn_like(latent)

        # ---------------------------------------------
        # DDPM reverse: x_T → x_0
        # ---------------------------------------------
        x = masked_latent + noisy * latent_mask  # start from noise

        # 6) DDPM reverse: x_T -> x_0
        for t in self.noise_scheduler.timesteps:
            # CondEncoder: masked_latents
            cond_tokens = self.cond_proj(masked_latent)

            # CoordEncoder: mask
            coord_encoder = self._require_coord_encoder()
            coord_tokens = coord_encoder(mask)

            # merge
            condition = torch.cat([cond_tokens, coord_tokens], dim=-1)  # (B,64,768)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(x, t, encoder_hidden_states=condition).sample

            # DDPM step
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample  # type: ignore

            # overwrite known region with original masked_latent
            x = latent_mask * x + (1 - latent_mask) * masked_latent

        # Decode image
        with torch.no_grad():
            image_recon = self.vae.decode(x / self.scaling_factor).sample  # type: ignore

        # convert to display format
        image_recon_disp = (image_recon.clamp(-1, 1) + 1) / 2
        result_np = image_recon_disp[0].permute(1, 2, 0).detach().cpu().numpy()

        plot_inpainting_triplet(
            image=self._to_numpy_image(image),
            mask=self._to_numpy_image(mask),
            result=result_np,
            figsize=(14, 4),
        )

        return x

    def run(self, cfg: InpaintRunConfig) -> list[torch.Tensor]:
        image_dir = Path(cfg.image_dir)
        mask_dir = Path(cfg.mask_dir)

        if not image_dir.exists() or not image_dir.is_dir():
            raise FileNotFoundError(f"image_dir is not a directory: {image_dir}")

        if not mask_dir.exists() or not mask_dir.is_dir():
            raise FileNotFoundError(f"mask_dir is not a directory: {mask_dir}")

        results: list[torch.Tensor] = []

        image_files = sorted(
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )

        for img_path in image_files:
            mask_path = mask_dir / img_path.name

            if not mask_path.exists():
                print(f"[Warning] No mask found for {img_path.name}, skipping.")
                continue

            res = self.run_one(cfg)
            results.append(res)

        return results
