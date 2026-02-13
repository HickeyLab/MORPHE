from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from disco.core.latent_diffusion.artifact import LatentDiffuserArtifact

from disco.core.latent_diffusion.infer.base import InferenceResult, LatentDiffusionInferencer
from disco.core.latent_diffusion.strategy.arbitrary_inpainting import ArbitraryInpainting
from disco.viz.decoded_img import plot_decoded_image, plot_inpainting_triplet


class ArbitraryInpaintingInferer(LatentDiffusionInferencer):
    
    REQUIRED_STRATEGY_NAME = "arbitrary_inpainting"

    def __init__(
        self, 
        *, 
        artifact: LatentDiffuserArtifact, 
        strategy: ArbitraryInpainting, 
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if getattr(strategy, "strategy_name", None) != self.REQUIRED_STRATEGY_NAME:
            raise TypeError(
                f"ArbitraryInpaintingInferer requires strategy_name='{self.REQUIRED_STRATEGY_NAME}', "
                f"got {getattr(strategy, 'strategy_name', None)!r}"
            )
        super().__init__(
            artifact=artifact, 
            strategy=strategy, 
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )

    def _validate_run_args(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        *,
        num_steps: int,
        plot_fig_size: tuple[int, int] | None,
    ) -> tuple[Path, Path]:
        if image_path is None or mask_path is None:
            raise RuntimeError(
                "ArbitraryInpaintingInferer requires image_path and mask_path but one of them is None."
            )

        image_path = Path(image_path)
        mask_path = Path(mask_path)

        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"image_path does not exist or is not a file: {image_path}")
        if not mask_path.exists() or not mask_path.is_file():
            raise FileNotFoundError(f"mask_path does not exist or is not a file: {mask_path}")

        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("`num_steps` must be a positive integer.")

        if self.vae is None or self.unet is None:
            raise RuntimeError("Inferencer requires loaded VAE and UNet.")
        if self.cond_proj is None or self.coord_encoder is None:
            raise RuntimeError("Inferencer requires loaded cond_proj and coord_encoder.")
        if self.noise_scheduler is None:
            raise RuntimeError("Inferencer requires loaded noise_scheduler.")
        
        if plot_fig_size is not None:
            if (
                not isinstance(plot_fig_size, tuple)
                or len(plot_fig_size) != 2
                or not all(isinstance(x, int) and x > 0 for x in plot_fig_size)
            ):
                raise ValueError(
                    "`plot_fig_size` must be tuple[int, int] with positive values."
                )

        return image_path, mask_path

    def _load_image(
        self, 
        path: str | Path
    ) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.strategy.img_size, self.strategy.img_size))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # (3,H,W)
        img = (img * 2 - 1)  # [-1, 1]
        return img.unsqueeze(0).to(self.device)  # (1,3,512,512)

    def _load_mask(self, path: str | Path) -> torch.Tensor:
        """
        mask: white=mask=1, black=keep=0
        """
        m = Image.open(path).convert("L").resize((self.strategy.img_size, self.strategy.img_size))
        m = np.array(m).astype(np.float32) / 255.0
        m = torch.tensor(m).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return m.to(self.device)

    def run_one(
        self,
        *,
        image_path: str | Path,
        mask_path: str | Path,
        num_steps: int = 200,
        show_plot: bool = True,
        plot_title: str | None = None,
        plot_fig_size: tuple[int, int] | None = None, 
    ) -> "InferenceResult": 
        image_path, mask_path = self._validate_run_args(
            image_path,
            mask_path,
            num_steps=num_steps,
            plot_fig_size=plot_fig_size,
        )

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        # 1. Encode image to latent
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample() * self.scaling_factor

        B, C, H, W = latent.shape

        # Expand mask to 4 latent channels
        latent_mask = F.interpolate(mask, size=(H, W), mode="nearest")
        latent_mask = latent_mask.expand(-1, C, -1, -1)

         # mask the latent
        masked_latent = latent * (1 - latent_mask)

        # show masked decode
        if show_plot:
            with torch.no_grad():
                m = self.vae.decode(masked_latent / self.scaling_factor).sample
            m = (m.clamp(-1, 1) + 1) / 2
            m = m[0].permute(1, 2, 0).detach().cpu().numpy()
            plot_decoded_image(
                preview=m,
                iteration=None,
                figsize=plot_fig_size,
                title=plot_title
            )

        # DDPM scheduler
        self.noise_scheduler.set_timesteps(num_steps, device=self.device)

        # Initial noise
        noisy = torch.randn_like(latent)
        
        # ---------------------------------------------
        # DDPM reverse: x_T → x_0
        # ---------------------------------------------
        x = masked_latent + noisy * latent_mask # start from noise

        # 6) DDPM reverse: x_T -> x_0
        for t in self.noise_scheduler.timesteps:
            # CondEncoder: masked_latents
            cond_tokens = self.cond_proj(masked_latent)
            
            # CoordEncoder: mask
            coord_tokens = self.coord_encoder(mask)

            # merge
            condition = torch.cat([cond_tokens, coord_tokens], dim=-1)  # (B,64,768)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(x, t, encoder_hidden_states=condition).sample

            # DDPM step
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample

            # overwrite known region with original masked_latent
            x = latent_mask * x + (1 - latent_mask) * masked_latent

        # Decode image
        with torch.no_grad():
            image_recon = self.vae.decode(x / self.scaling_factor).sample

        # convert to display format
        image_recon_disp = (image_recon.clamp(-1, 1) + 1) / 2
        result_np = image_recon_disp[0].permute(1, 2, 0).detach().cpu().numpy()
        
        plot_inpainting_triplet(
            image=(image[0].permute(1,2,0).cpu()+1)/2,
            mask=mask[0,0].cpu(),
            result=result_np,
            figsize=(14,4)
        )

        return InferenceResult(
            image=image_recon,
            latents=x,
            extras={
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "num_steps": int(num_steps),
                "result_numpy": result_np,
                "input_numpy": ((image[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2),
                "mask_numpy": mask[0, 0].detach().cpu().numpy(),
            },
        )
