from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Self

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel # type: ignore

from ..artifact import LatentDiffusionArtifact
from .base import BaseLatentInferencer
from .run_config import ThreeDimImputationRunConfig, ThreeDimImputationWeightSweepConfig
from ..model import CondEncoder3D
from ....constants import InferenceMode
from ....utils import resolve_device


class ThreeDimImputationInferencer(BaseLatentInferencer):
    """
    Inferencer for three-dimensional latent imputation between two endpoint images.

    This inferencer encodes a previous and next image into latent space, blends
    them according to user-provided weights, conditions the diffusion model on
    that blended latent representation, and generates an intermediate latent and
    decoded image.
    """

    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        cond_encoder: CondEncoder3D,
        coord_encoder: None,
        bbox_encoder: None,
        noise_scheduler: DDPMScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize a three-dimensional imputation inferencer.

        Args:
            artifact: Trained latent diffusion artifact for three-dimensional
                imputation.
            device: Device to run inference on. If ``None``, a default device
                is resolved by the base inferencer.
            dtype: Floating-point dtype to use for inference. If ``None``,
                a default dtype is resolved by the base inferencer.

        Raises:
            RuntimeError: If the artifact was not created for
                three-dimensional imputation.
        """
        if artifact.inference_mode != InferenceMode.THREE_DIMENSIONAL_IMPUTATION:
            raise RuntimeError(
                "ThreeDimImputationInferencer requires an artifact with "
                f"inference_mode={InferenceMode.THREE_DIMENSIONAL_IMPUTATION}, "
                f"but got {artifact.inference_mode}."
            )
            
        if cond_encoder is None or not isinstance(cond_encoder, CondEncoder3D):
            raise RuntimeError(
                "ThreeDimImputationInferencer requires cond_encoder to be present in the artifact."
            )

        super().__init__(
            artifact=artifact,
            unet=unet,
            vae=vae,
            cond_encoder=cond_encoder,
            coord_encoder=coord_encoder,
            bbox_encoder=bbox_encoder,
            noise_scheduler=noise_scheduler,
            device=device,
            dtype=dtype,
        )

        self.transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        
    @classmethod
    def from_artifact(
        cls,
        artifact: LatentDiffusionArtifact,
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        """
        Construct a three-dimensional imputation inferencer from a latent diffusion artifact.

        Args:
            artifact: Latent diffusion artifact containing model weights and
                architecture for three-dimensional imputation.
            device: Device to run inference on. If ``None``, a default device
                is resolved.
        """
        if artifact.inference_mode != InferenceMode.THREE_DIMENSIONAL_IMPUTATION:
            raise RuntimeError(
                "ThreeDimImputationInferencer requires an artifact with "
                f"inference_mode={InferenceMode.THREE_DIMENSIONAL_IMPUTATION}, "
                f"but got {artifact.inference_mode}."
            )

        resolved_device = resolve_device(device)
        resolved_dtype = torch.float32

        unet, vae, cond_encoder, _, bbox_encoder, noise_scheduler = (
            artifact.architecture.build_components(
                device=resolved_device,
                dtype=resolved_dtype,
            )
        )
        
        if not isinstance(cond_encoder, CondEncoder3D):
            raise RuntimeError(
                "ThreeDimImputationInferencer requires a CondEncoder3D in the artifact architecture."
            )

        return cls(
            artifact=artifact,
            unet=unet,
            vae=vae,
            cond_encoder=cond_encoder,
            coord_encoder=None,
            bbox_encoder=None,
            noise_scheduler=noise_scheduler,
            device=resolved_device,
            dtype=resolved_dtype,
        )

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert an image tensor from ``[-1, 1]`` into ``[0, 1]``.

        Args:
            x: Tensor image data in normalized diffusion/VAE output space.

        Returns:
            The denormalized tensor.
        """
        return (x.clamp(-1, 1) + 1) / 2

    def _load_image_tensor(self, path: str | Path) -> torch.Tensor:
        """
        Load, normalize, batch, and move an image onto the inferencer device.

        Args:
            path: Path to the input image file.

        Returns:
            A tensor of shape ``(1, 3, H, W)`` on the inferencer device and dtype.
        """
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = self.transform(rgb_image).unsqueeze(0)

        return image_tensor.to(device=self.device, dtype=self.dtype)

    def _save_outputs(
        self,
        *,
        latents: torch.Tensor,
        decoded: torch.Tensor,
        save_dir: str | Path,
        save_name: str,
        save_latents_name: str,
        save_png_name: str,
    ) -> None:
        """
        Save the generated latent tensor and decoded PNG output.

        Args:
            latents: Generated latent tensor to serialize.
            decoded: Decoded image tensor in model output space.
            save_dir: Root directory where outputs should be written.
            save_name: Subdirectory name under ``save_dir``.
            save_latents_name: Filename for the saved latent tensor.
            save_png_name: Filename for the saved PNG image.
        """
        output_dir = Path(save_dir) / save_name
        output_dir.mkdir(parents=True, exist_ok=True)

        latents_to_save = latents.detach().cpu()
        torch.save(latents_to_save, output_dir / save_latents_name)

        image_tensor = decoded.detach().cpu()
        image_tensor = self._denormalize(image_tensor)
        image_tensor = image_tensor[0].permute(1, 2, 0)

        if image_tensor.max() <= 1.0:
            image_uint8 = (image_tensor * 255.0).round().clamp(0, 255).to(torch.uint8)
        else:
            image_uint8 = image_tensor.round().clamp(0, 255).to(torch.uint8)

        Image.fromarray(image_uint8.numpy()).save(
            output_dir / save_png_name,
            format="PNG",
        )

    @torch.no_grad()
    def _run_one_from_config(self, cfg: ThreeDimImputationRunConfig) -> torch.Tensor:
        """
        Run a single three-dimensional imputation inference pass.

        This method loads the previous and next images, encodes both into latent
        space, builds a weighted conditioning representation, performs reverse
        diffusion, decodes the resulting latent, and saves both latent and image
        outputs.

        Args:
            cfg: Configuration for a single three-dimensional imputation run.

        Returns:
            The generated intermediate latent tensor.
        """
        img_prev = self._load_image_tensor(cfg.prev_img_path)
        img_next = self._load_image_tensor(cfg.next_img_path)

        latent_prev = self.encode_with_vae(img_prev)
        latent_next = self.encode_with_vae(img_next)

        w_prev = torch.tensor(cfg.w_prev, device=self.device, dtype=latent_prev.dtype)
        w_next = torch.tensor(cfg.w_next, device=self.device, dtype=latent_next.dtype)
        condition = self.cond_encoder(w_prev * latent_prev + w_next * latent_next)

        self.noise_scheduler.set_timesteps(
            cfg.num_steps,
            device=self.device,
        )

        latents = torch.randn_like(latent_prev)

        for timestep in tqdm(self.noise_scheduler.timesteps, leave=False):
            noise_pred = self.unet(
                latents,
                timestep,
                encoder_hidden_states=condition,
            ).sample

            latents = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timestep,  # type: ignore[arg-type]
                sample=latents,
            ).prev_sample  # type: ignore[attr-defined]

        decoded = self.vae.decode(latents / self.scaling_factor).sample  # type: ignore[attr-defined]

        self._save_outputs(
            latents=latents / self.scaling_factor,
            decoded=decoded,
            save_dir=cfg.save_dir,
            save_name=cfg.save_name,
            save_latents_name=cfg.save_latents_name,
            save_png_name=cfg.save_png_name,
        )

        return latents / self.scaling_factor

    @torch.no_grad()
    def run(
        self,
        prev_img_path: str | Path,
        next_img_path: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        num_steps: int = 200,
        w_prev: float = 0.5,
        w_next: float = 0.5,
        save_latents_name: str = "latents.pt",
        save_png_name: str = "output.png",
    ) -> torch.Tensor:
        """
        Run three-dimensional imputation for one previous/next image pair.

        This is the public single-run entry point. It builds a
        ``ThreeDimImputationRunConfig`` and delegates execution to the shared
        internal config-driven implementation.

        Args:
            prev_img_path: Path to the previous image.
            next_img_path: Path to the next image.
            save_dir: Root directory where outputs should be written.
            save_name: Subdirectory under ``save_dir`` for this run's outputs.
            num_steps: Number of denoising steps.
            w_prev: Weight applied to the previous-image latent.
            w_next: Weight applied to the next-image latent.
            save_latents_name: Output filename for the saved latent tensor.
            save_png_name: Output filename for the saved PNG image.

        Returns:
            The generated intermediate latent tensor.
        """
        cfg = ThreeDimImputationRunConfig(
            prev_img_path=prev_img_path,
            next_img_path=next_img_path,
            save_dir=save_dir,
            save_name=save_name,
            num_steps=num_steps,
            w_prev=w_prev,
            w_next=w_next,
            save_latents_name=save_latents_name,
            save_png_name=save_png_name,
        )
        return self._run_one_from_config(cfg)

    @torch.no_grad()
    def run_weight_sweep(
        self,
        prev_img_path: str | Path,
        next_img_path: str | Path,
        save_dir: str | Path = Path("./outputs"),
        num_steps: int = 200,
        start: float = 0.1,
        end: float = 0.9,
        step: float = 0.1,
    ) -> list[torch.Tensor]:
        """
        Run three-dimensional imputation across a sweep of interpolation weights.

        For each weight ``w_prev`` in the requested range, this method sets
        ``w_next = 1 - w_prev`` and runs a full imputation pass, saving uniquely
        named latent and PNG outputs for each weight pair.

        Args:
            prev_img_path: Path to the previous image.
            next_img_path: Path to the next image.
            save_dir: Directory where outputs should be written.
            num_steps: Number of denoising steps per run.
            start: Starting value for ``w_prev``.
            end: Ending value for ``w_prev``.
            step: Increment for ``w_prev``.

        Returns:
            A list of generated intermediate latent tensors, one per sweep point.
        """

        cfg = ThreeDimImputationWeightSweepConfig(
            prev_img_path=prev_img_path,
            next_img_path=next_img_path,
            save_dir=save_dir,
            num_steps=num_steps,
            start=start,
            end=end,
            step=step,
        )

        results: list[torch.Tensor] = []
        weight = cfg.start

        while weight <= cfg.end + 1e-8:
            w_prev = round(weight, 6)
            w_next = 1.0 - w_prev

            lat_name = f"{w_prev:.1f}_{w_next:.1f}.pt"
            png_name = f"mid_{w_prev:.3f}_{w_next:.3f}.png"

            run_cfg = ThreeDimImputationRunConfig(
                prev_img_path=cfg.prev_img_path,
                next_img_path=cfg.next_img_path,
                save_dir=cfg.save_dir,
                num_steps=cfg.num_steps,
                w_prev=w_prev,
                w_next=w_next,
                save_latents_name=lat_name,
                save_png_name=png_name,
            )
            results.append(self._run_one_from_config(run_cfg))

            weight += cfg.step

        return results