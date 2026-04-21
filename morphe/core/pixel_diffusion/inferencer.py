from __future__ import annotations

from pathlib import Path
from typing import Self

import torch
from diffusers import DDPMScheduler  # type: ignore
from torchvision import transforms

from .artifact import PixelDiffusionArtifact
from .models import LatentAdapter, UNet512
from ...utils import resolve_device, resolve_dtype


class PixelDiffusionInferencer:
    """
    Inference wrapper for the pixel diffusion stage.

    This class reconstructs the trained pixel diffusion runtime from a
    ``PixelDiffusionArtifact`` and provides convenience methods for decoding
    latent tensors into image tensors or saved PNG files.

    The inferencer is intended for inference-only usage and keeps its modules
    in evaluation mode.
    """
    def __init__(
        self,
        *,
        artifact: PixelDiffusionArtifact,
        adapter: LatentAdapter,
        unet512: UNet512,
        noise_scheduler: DDPMScheduler,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(artifact, PixelDiffusionArtifact):
            raise TypeError(
                f"Expected artifact of type PixelDiffusionArtifact, got {type(artifact)}."
            )
        
        if not adapter or not isinstance(adapter, LatentAdapter):
            raise ValueError(
                f"Latent adapter is required for PixelDiffusionInferencer and must be a LatentAdapter instance."
            )
        
        if not unet512 or not isinstance(unet512, UNet512):
            raise ValueError(
                f"UNet512 model is required for PixelDiffusionInferencer and must be a UNet512 instance."
            )
        
        if not noise_scheduler or not isinstance(noise_scheduler, DDPMScheduler):
            raise ValueError(
                f"Noise scheduler is required for PixelDiffusionInferencer and must be a DDPMScheduler instance."
            )
            
        self.artifact = artifact
        self.adapter = adapter
        self.unet512 = unet512
        self.noise_scheduler = noise_scheduler
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(device=self.device, dtype=dtype)
        self._to_pil = transforms.ToPILImage()

        self.adapter.eval()
        self.unet512.eval()

    @classmethod
    def from_artifact(
        cls,
        artifact: PixelDiffusionArtifact,
        *,
        scheduler_num_inference_steps: int = 150,
        device: torch.device | str | None = None,
    ) -> Self:
        adapter, unet512, noise_scheduler, resolved_device, resolved_dtype = (
            artifact.build_inference_components(
                device=device,
                dtype=torch.float32,
                scheduler_num_inference_steps=scheduler_num_inference_steps
            )
        )

        return cls(
            artifact=artifact,
            adapter=adapter,
            unet512=unet512,
            noise_scheduler=noise_scheduler,
            device=resolved_device,
            dtype=resolved_dtype,
        )

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
        Decode a latent into an image tensor.

        Args:
            latent: Latent tensor with shape [4, 64, 64] or [B, 4, 64, 64].
            latent_path: Path to a torch-saved latent tensor. Mutually exclusive
                with ``latent``.
            seed: Optional random seed for reproducible decoding noise.
            out_h: Output image height.
            out_w: Output image width.

        Returns:
            Image tensor with shape [3, H, W] in [0, 1] on CPU.
        """
        self._validate_decode_inputs(latent=latent, latent_path=latent_path)

        z = self._load_latent(latent=latent, latent_path=latent_path)
        z = self._ensure_batched_latent(z).to(device=self.device, dtype=self.dtype)

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        cond_feats = self.adapter(z)

        batch_size = z.shape[0]
        x = torch.randn(
            batch_size,
            3,
            out_h,
            out_w,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        ) * self.noise_scheduler.init_noise_sigma

        for timestep in self.noise_scheduler.timesteps:
            timestep_batch = torch.full(
                (batch_size,),
                int(timestep),
                device=self.device,
                dtype=torch.long,
            )
            x0_pred = self.unet512(x, timestep_batch, cond_feats)
            x = self.noise_scheduler.step(x0_pred, timestep, x).prev_sample  # type: ignore

        output = (x.clamp(-1, 1) + 1) / 2
        return output[0].float().cpu()

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
        Decode a latent and save the result as a PNG file.

        Returns:
            Path to the saved PNG.
        """
        if not output_dir:
            raise ValueError("output_dir must be provided.")

        image = self.decode(
            latent=latent,
            latent_path=latent_path,
            seed=seed,
            out_h=out_h,
            out_w=out_w,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / output_name
        pil_image = self._to_pil(image.clamp(0, 1))
        pil_image.save(output_path)

        return output_path

    @staticmethod
    def _validate_decode_inputs(
        *,
        latent: torch.Tensor | None,
        latent_path: str | Path | None,
    ) -> None:
        """Validate mutually exclusive latent input arguments."""
        if latent is None and latent_path is None:
            raise ValueError("Provide either `latent` or `latent_path`.")
        if latent is not None and latent_path is not None:
            raise ValueError("Provide only one of `latent` or `latent_path` (not both).")
        if latent_path is not None:
            latent_path = Path(latent_path)
            if not latent_path.exists():
                raise FileNotFoundError(f"latent_path does not exist: {latent_path}")

    @staticmethod
    def _load_latent(
        *,
        latent: torch.Tensor | None,
        latent_path: str | Path | None,
    ) -> torch.Tensor:
        """Return a latent tensor from an in-memory tensor or a saved file."""
        if latent is not None:
            if not isinstance(latent, torch.Tensor):
                raise TypeError("latent must be a torch.Tensor.")
            return latent
        
        if latent_path is None:
            raise ValueError("latent_path must be provided if latent is None.")
        latent_path = Path(latent_path)
        loaded_latent = torch.load(latent_path, map_location="cpu")
        
        if not isinstance(loaded_latent, torch.Tensor):
            raise TypeError("Loaded latent is not a torch.Tensor.")
        return loaded_latent

    @staticmethod
    def _ensure_batched_latent(latent: torch.Tensor) -> torch.Tensor:
        """Ensure the latent has shape [B, C, H, W]."""
        if latent.ndim == 3:
            return latent.unsqueeze(0)
        if latent.ndim != 4:
            raise ValueError(
                f"latent must be [4,64,64] or [B,4,64,64], got ndim={latent.ndim}."
            )
        return latent