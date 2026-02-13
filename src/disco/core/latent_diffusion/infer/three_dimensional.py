from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from disco.core.latent_diffusion.artifact import LatentDiffuserArtifact

from disco.core.latent_diffusion.infer.base import InferenceResult, LatentDiffusionInferencer
from disco.core.latent_diffusion.strategy.three_dimensional_imputation import ThreeDimensionalImputation


# -------------------------------------------------
# Utilities
# -------------------------------------------------



class ThreeDimensionalInferer(LatentDiffusionInferencer):
    REQUIRED_STRATEGY_NAME = "slice3d"

    def __init__(
        self, 
        *, 
        artifact: LatentDiffuserArtifact, 
        strategy: ThreeDimensionalImputation, 
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if getattr(strategy, "strategy_name", None) != self.REQUIRED_STRATEGY_NAME:
            raise TypeError(
                f"ThreeDimensionalInferer requires strategy_name='{self.REQUIRED_STRATEGY_NAME}', "
                f"got {getattr(strategy, 'strategy_name', None)!r}"
            )
        super().__init__(
            artifact=artifact, 
            strategy=strategy, 
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _validate_run_args(
        self,
        prev_path: str | Path,
        next_path: str | Path,
        out_dir: str | Path,
        *,
        num_inference_steps: int,
        w_prev: float,
        w_next: float,
        save_latents_name: str,
        save_png_name: str,
    ) -> tuple[Path, Path, Path]:
        if prev_path is None or next_path is None or out_dir is None:
            raise RuntimeError("ThreeDimensionalInferer requires prev_path, next_path, and out_dir but one is None.")

        prev_path = Path(prev_path)
        next_path = Path(next_path)
        out_dir = Path(out_dir)

        if not prev_path.exists() or not prev_path.is_file():
            raise FileNotFoundError(f"prev_path does not exist or is not a file: {prev_path}")
        if not next_path.exists() or not next_path.is_file():
            raise FileNotFoundError(f"next_path does not exist or is not a file: {next_path}")

        out_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("`num_inference_steps` must be a positive integer.")

        if not isinstance(w_prev, (int, float)):
            raise ValueError("`w_prev` must be a float.")
        if not isinstance(w_next, (int, float)):
            raise ValueError("`w_next` must be a float.")
        if abs((float(w_prev) + float(w_next)) - 1.0) > 1e-6:
            raise ValueError("`w_prev + w_next` must equal 1.0 (within tolerance).")

        if not isinstance(save_latents_name, str) or not save_latents_name:
            raise ValueError("`save_latents_name` must be a non-empty string.")
        if not isinstance(save_png_name, str) or not save_png_name:
            raise ValueError("`save_png_name` must be a non-empty string.")

        if self.vae is None or self.unet is None:
            raise RuntimeError("Inferencer requires loaded VAE and UNet.")
        if self.cond_proj is None:
            raise RuntimeError("Inferencer requires loaded cond_proj.")
        if self.noise_scheduler is None:
            raise RuntimeError("Inferencer requires loaded noise_scheduler.")

        return prev_path, next_path, out_dir
    
    def _denormalize(x):
        # [-1,1] -> [0,1]
        return (x.clamp(-1, 1) + 1) / 2

    @torch.no_grad()
    def run_one(
        self,
        *,
        prev_path: str | Path,
        next_path: str | Path,
        out_dir: str | Path,
        num_inference_steps: int = 200,
        w_prev: float = 0.5,
        w_next: float = 0.5,
        save_latents_name: str,
        save_png_name: str,
    ) -> "InferenceResult":
        prev_path, next_path, out_dir = self._validate_run_args(
            prev_path,
            next_path,
            out_dir,
            num_inference_steps=num_inference_steps,
            w_prev=w_prev,
            w_next=w_next,
            save_latents_name=save_latents_name,
            save_png_name=save_png_name,
        )

        # ====== load & preprocess images ======
        img_prev = Image.open(prev_path).convert("RGB")
        img_next = Image.open(next_path).convert("RGB")

        img_prev = self.transform(img_prev)
        img_next = self.transform(img_next)

        img_prev = img_prev.unsqueeze(0).to(self.device)
        img_next = img_next.unsqueeze(0).to(self.device)

        self.vae.eval()
        self.unet.eval()
        self.cond_proj.eval()

        # --------------------------------------------------
        # 1. Encode prev / next
        # --------------------------------------------------
        latent_prev = self.vae.encode(img_prev).latent_dist.sample()
        latent_next = self.vae.encode(img_next).latent_dist.sample()

        latent_prev = latent_prev * self.scaling_factor
        latent_next = latent_next * self.scaling_factor

        # --------------------------------------------------
        # 2. Build condition (USE NEXT)
        # --------------------------------------------------
        wp = torch.tensor(w_prev, device=self.device)
        wn = torch.tensor(w_next, device=self.device)
        condition = self.cond_proj(wp * latent_prev + wn * latent_next)

        # --------------------------------------------------
        # 3. Setup scheduler & timesteps
        # --------------------------------------------------
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # --------------------------------------------------
        # 4. Add noise to prev latent (img2img start)
        # --------------------------------------------------
        latents = torch.randn_like(latent_prev)

        # --------------------------------------------------
        # 5. Reverse diffusion
        # --------------------------------------------------
        for t in tqdm(timesteps, leave=False):
            x0_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=condition
            ).sample

            latents = self.noise_scheduler.step(
                model_output=x0_pred,
                timestep=t,
                sample=latents
            ).prev_sample

        # --------------------------------------------------
        # 6. Decode
        # --------------------------------------------------
        latents = latents / self.scaling_factor
        torch.save(latents, str(Path(out_dir) / save_latents_name))
        img_mid_pred = self.vae.decode(latents).sample

        x = img_mid_pred.detach().cpu()
        x = self._denormalize(x)
        x = x[0].permute(1, 2, 0)

        if x.max() <= 1.0:
            x_uint8 = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
        else:
            x_uint8 = x.round().clamp(0, 255).to(torch.uint8)

        x_np = x_uint8.numpy()

        Image.fromarray(x_np).save(
            str(Path(out_dir) / save_png_name),
            format="PNG"
        )

        return InferenceResult(
            image=img_mid_pred,
            latents=latents,
            extras={
                "prev_path": str(prev_path),
                "next_path": str(next_path),
                "out_dir": str(out_dir),
                "num_inference_steps": int(num_inference_steps),
                "w_prev": float(w_prev),
                "w_next": float(w_next),
                "latents_save_path": str(Path(out_dir) / save_latents_name),
                "png_save_path": str(Path(out_dir) / save_png_name),
            },
        )
