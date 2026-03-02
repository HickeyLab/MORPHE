from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from diffusers import DDPMScheduler
from torchvision import transforms

from disco.core.pixel_diffusion.artifact import PixelDiffusionTrainerArtifact


@dataclass(frozen=True)
class PixelDiffusionRuntime:
    adapter: torch.nn.Module
    unet512: torch.nn.Module
    noise_scheduler: DDPMScheduler
    device: torch.device
    dtype: torch.dtype


class PixelDiffusionInferencer:
    """
    Pixel diffuser inferencer (Cascade 512 stage).

    Mirrors your LatentDiffuserInferencer style:
      - __init__ builds a runtime from the artifact
      - inference methods just use self.adapter/self.unet512/self.noise_scheduler
    """

    def __init__(
        self,
        *,
        artifact: PixelDiffusionTrainerArtifact,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        num_inference_steps: int = 150,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        rt: PixelDiffusionRuntime = artifact.build_inference_runtime(
            pretrained_path=pretrained_path,
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
        )

        self.adapter = rt.adapter
        self.unet512 = rt.unet512
        self.noise_scheduler = rt.noise_scheduler
        self.device = rt.device
        self.dtype = rt.dtype

        self._to_pil = transforms.ToPILImage()

    @torch.no_grad()
    def decode(
        self,
        *,
        latent: torch.Tensor | None = None,
        latent_path: str | Path | None = None,
        seed: int | None = None,
        out_h: int = 512,
        out_w: int = 512,
    ) -> torch.Tensor:
        """
        Decode a latent into an image.

        Inputs:
          - latent: tensor [4,64,64] or [B,4,64,64]
          - latent_path: path to torch-saved tensor (mutually exclusive with latent)

        Returns:
          - image tensor [3,H,W] in [0,1] on CPU
        """
        self._validate_decode_inputs(latent=latent, latent_path=latent_path)

        z = self._load_latent(latent=latent, latent_path=latent_path)
        z = self._normalize_latent_shape(z).to(device=self.device, dtype=self.dtype)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        cond_feats = self.adapter(z)

        B = z.shape[0]
        x = torch.randn(
            B,
            3,
            out_h,
            out_w,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        ) * self.noise_scheduler.init_noise_sigma

        for t in self.noise_scheduler.timesteps:
            t_batch = torch.full((B,), int(t), device=self.device, dtype=torch.long)
            x0_pred = self.unet512(x, t_batch, cond_feats)
            x = self.noise_scheduler.step(x0_pred, t, x).prev_sample

        out = (x.clamp(-1, 1) + 1) / 2  # [B,3,H,W] in [0,1]
        return out[0].float().cpu()

    def decode_to_png(
        self,
        *,
        latent: torch.Tensor | None = None,
        latent_path: str | Path | None = None,
        output_dir: str | Path,
        output_name: str = "decoded_from_latent.png",
        seed: int | None = None,
        out_h: int = 512,
        out_w: int = 512,
    ) -> Path:
        """
        Decode and save as PNG. Returns the saved file path.
        """
        if not output_dir:
            raise ValueError("output_dir must be provided.")

        img = self.decode(
            latent=latent,
            latent_path=latent_path,
            seed=seed,
            out_h=out_h,
            out_w=out_w,
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / output_name
        pil = self._to_pil(img.clamp(0, 1))
        pil.save(str(out_path))
        return out_path

    @staticmethod
    def _validate_decode_inputs(
        *,
        latent: torch.Tensor | None,
        latent_path: str | Path | None,
    ) -> None:
        if latent is None and latent_path is None:
            raise ValueError("Provide either `latent` or `latent_path`.")
        if latent is not None and latent_path is not None:
            raise ValueError("Provide only one of `latent` or `latent_path` (not both).")
        if latent_path is not None:
            p = Path(latent_path)
            if not p.exists():
                raise FileNotFoundError(f"latent_path does not exist: {p}")

    @staticmethod
    def _load_latent(
        *,
        latent: torch.Tensor | None,
        latent_path: str | Path | None,
    ) -> torch.Tensor:
        if latent is not None:
            if not isinstance(latent, torch.Tensor):
                raise TypeError("latent must be a torch.Tensor.")
            return latent

        z = torch.load(str(latent_path), map_location="cpu")
        if not isinstance(z, torch.Tensor):
            raise TypeError("Loaded latent is not a torch.Tensor.")
        return z

    @staticmethod
    def _normalize_latent_shape(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            return z.unsqueeze(0)  # [4,64,64] -> [1,4,64,64]
        if z.ndim != 4:
            raise ValueError(f"latent must be [4,64,64] or [B,4,64,64], got ndim={z.ndim}.")
        return z