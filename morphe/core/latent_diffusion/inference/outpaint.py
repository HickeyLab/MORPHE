from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel # type: ignore

from core.latent_diffusion.artifact import LatentDiffusionArtifact
from core.latent_diffusion.inference.base import BaseLatentInferencer
from core.latent_diffusion.inference.run_config import OutpaintRunConfig
from core.latent_diffusion.model import BBoxEncoder, CondEncoder
from constants import InferenceMode
from utils import resolve_device, resolve_dtype
from viz.decoded_img import plot_decoded_image

Direction = Literal["right", "left", "down", "up"]

class OutpaintInferencer(BaseLatentInferencer):
    """
    Inference wrapper for latent diffusion outpainting.

    This inferencer expands an image outward in one direction by repeatedly
    preserving the existing region, generating a new border region, stitching
    that region onto the accumulated image, and cyclically shifting the latest
    output to prepare the next iteration.
    """

    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        cond_encoder: CondEncoder,
        coord_encoder: None,
        bbox_encoder: BBoxEncoder,
        noise_scheduler: DDPMScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize an outpainting inferencer from a trained latent artifact.

        Args:
            artifact: Trained latent diffusion artifact for outpainting.
            device: Device to run inference on. If ``None``, a default device
                is resolved by the base inferencer.
            dtype: Floating-point dtype to use for inference. If ``None``,
                a default dtype is resolved by the base inferencer.

        Raises:
            RuntimeError: If the artifact was not created for outpainting,
                if the bbox encoder is missing, or if ``artifact.img_size``
                is not defined.
        """
        if artifact.inference_mode != InferenceMode.OUTPAINTING:
            raise RuntimeError(
                "OutpaintInferencer requires an artifact with "
                f"inference_mode={InferenceMode.OUTPAINTING}, "
                f"but got {artifact.inference_mode}."
            )

        if bbox_encoder is None:
            raise RuntimeError(
                "OutpaintInferencer requires bbox_encoder to be present in the artifact."
            )

        if artifact.img_size is None:
            raise RuntimeError(
                "OutpaintInferencer requires artifact.img_size to be defined."
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
                transforms.Resize((artifact.img_size, artifact.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        
    @classmethod
    def from_artifact(
        cls,
        artifact: LatentDiffusionArtifact,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """
        Construct an ``OutpaintInferencer`` from a latent diffusion artifact.

        This method validates that the artifact is compatible with outpainting
        inference and extracts the necessary components to initialize the
        inferencer.

        Args:
            artifact: Latent diffusion artifact containing trained model states
                and architecture specifications.
            device: Target device for inference. If ``None``, a default device is
                resolved.
            dtype: Target floating-point dtype for inference. If ``None``, a
                default dtype is resolved.

        Returns:
            An initialized ``OutpaintInferencer`` instance.
        """
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(resolved_device, dtype)

        unet, vae, cond_encoder, _, bbox_encoder, noise_scheduler = (
            artifact.architecture.build_components(
                device=resolved_device,
                dtype=resolved_dtype,
            )
        )

        if cond_encoder is None or not isinstance(cond_encoder, CondEncoder):
            raise RuntimeError(
                "OutpaintInferencer requires a cond encoder in the artifact architecture."
            )

        if bbox_encoder is None or not isinstance(bbox_encoder, BBoxEncoder):
            raise RuntimeError(
                "OutpaintInferencer requires a bbox encoder in the artifact architecture."
            )

        return cls(
            artifact=artifact,
            unet=unet,
            vae=vae,
            cond_encoder=cond_encoder,
            coord_encoder=None,
            bbox_encoder=bbox_encoder,
            noise_scheduler=noise_scheduler,
            device=resolved_device,
            dtype=resolved_dtype,
        )

    def _load_image_tensor(self, image_path: str | Path) -> torch.Tensor:
        """
        Load, normalize, and batch an input image for outpainting.

        Args:
            image_path: Path to the input image file.

        Returns:
            A normalized tensor of shape ``(1, 3, H, W)`` on the inferencer
            device and dtype.
        """
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = self.transform(rgb_image).unsqueeze(0)

        return image_tensor.to(device=self.device, dtype=self.dtype)

    def _require_bbox_encoder(self) -> torch.nn.Module:
        """
        Return the bbox encoder required for outpainting inference.

        Returns:
            The bbox encoder stored on the inferencer.

        Raises:
            RuntimeError: If the bbox encoder is not available.
        """
        if self.bbox_encoder is None:
            raise RuntimeError("bbox_encoder is required for outpaint inference.")
        return self.bbox_encoder

    def _resolve_bbox(self, cfg: OutpaintRunConfig) -> torch.Tensor:
        """
        Resolve the normalized bbox corresponding to the outpaint region.

        The bbox marks the region to be newly generated in the current frame.

        Args:
            cfg: Outpainting run configuration.

        Returns:
            A bbox tensor of shape ``(1, 4)`` in normalized coordinates
            ``[x1, y1, x2, y2]``.
        """
        if cfg.direction == "right":
            return torch.tensor(
                [[cfg.crop_ratio, 0.0, 1.0, 1.0]],
                device=self.device,
                dtype=self.dtype,
            )
        elif cfg.direction == "left":
            return torch.tensor(
                [[0.0, 0.0, 1.0 - cfg.crop_ratio, 1.0]],
                device=self.device,
                dtype=self.dtype,
            )
        elif cfg.direction == "down":
            return torch.tensor(
                [[0.0, cfg.crop_ratio, 1.0, 1.0]],
                device=self.device,
                dtype=self.dtype,
            )
        elif cfg.direction == "up":    
            return torch.tensor(
                [[0.0, 0.0, 1.0, 1.0 - cfg.crop_ratio]],
                device=self.device,
                dtype=self.dtype,
            )
            
        else:
            raise ValueError(f"Invalid direction {cfg.direction!r} in run config.")

    def _create_image_mask(
        self,
        bbox: torch.Tensor,
        image_shape: Sequence[int],
    ) -> torch.Tensor:
        """
        Build a binary image-space mask from normalized bounding boxes.

        Args:
            bbox: Tensor of normalized bounding boxes with shape ``(N, 4)``.
            image_shape: Shape of the image tensor, typically ``(B, C, H, W)``.

        Returns:
            A mask tensor of shape ``(N, 1, H, W)`` on the inferencer device,
            with value ``1`` inside the bbox region and ``0`` elsewhere.
        """
        _, _, height, width = image_shape
        masks: list[torch.Tensor] = []

        x_coords, y_coords = torch.meshgrid(
            torch.arange(width, device=self.device),
            torch.arange(height, device=self.device),
            indexing="xy",
        )

        for coords in bbox:
            x1 = coords[0] * width
            y1 = coords[1] * height
            x2 = coords[2] * width
            y2 = coords[3] * height

            mask = (
                (x_coords >= x1)
                & (x_coords <= x2)
                & (y_coords >= y1)
                & (y_coords <= y2)
            ).float()
            masks.append(mask)

        return torch.stack(masks, dim=0).unsqueeze(1)

    def _create_latent_mask(
        self,
        bbox: torch.Tensor,
        latent_shape: Sequence[int],
    ) -> torch.Tensor:
        """
        Build a binary latent-space mask from normalized bounding boxes.

        Args:
            bbox: Tensor of normalized bounding boxes with shape ``(N, 4)``.
            latent_shape: Shape of the latent tensor, typically
                ``(B, C, H_latent, W_latent)``.

        Returns:
            A mask tensor of shape ``(N, 1, H_latent, W_latent)`` on the
            inferencer device, with value ``1`` inside the bbox region.
        """
        _, _, latent_height, latent_width = latent_shape
        masks: list[torch.Tensor] = []

        x_coords, y_coords = torch.meshgrid(
            torch.arange(latent_width, device=self.device),
            torch.arange(latent_height, device=self.device),
            indexing="xy",
        )

        for coords in bbox:
            x1 = coords[0] * latent_width
            y1 = coords[1] * latent_height
            x2 = coords[2] * latent_width
            y2 = coords[3] * latent_height

            mask = (
                (x_coords >= x1)
                & (x_coords <= x2)
                & (y_coords >= y1)
                & (y_coords <= y2)
            ).float()
            masks.append(mask)

        return torch.stack(masks, dim=0).unsqueeze(1)

    def _extract_old_region(
        self,
        image_tensor: torch.Tensor,
        direction: str,
        crop_ratio: float,
    ) -> torch.Tensor:
        """
        Extract the preserved region from the current image.

        Args:
            image_tensor: Input tensor of shape ``(1, 3, H, W)``.
            direction: Outpainting direction.
            crop_ratio: Fraction of the image to preserve before generating
                the new border region.

        Returns:
            The preserved image region tensor.
        """
        _, _, height, width = image_tensor.shape

        if direction == "right":
            preserve_width = int(width * crop_ratio)
            return image_tensor[..., :, :preserve_width]

        elif direction == "left":
            preserve_width = int(width * crop_ratio)
            return image_tensor[..., :, width - preserve_width :]

        elif direction == "down":
            preserve_height = int(height * crop_ratio)
            return image_tensor[..., :preserve_height, :]
        
        elif direction == "up":
            preserve_height = int(height * crop_ratio)
            return image_tensor[..., height - preserve_height :, :]

        else:
            raise ValueError(f"Invalid direction {direction!r} for extracting old region.")

    def _extract_new_region(
        self,
        generated: torch.Tensor,
        direction: str,
        patch_size: int,
    ) -> torch.Tensor:
        """
        Extract the newly generated border patch from a decoded output image.

        Args:
            generated: Generated image tensor of shape ``(1, 3, H, W)``.
            direction: Outpainting direction.
            patch_size: Width or height of the newly generated border region.

        Returns:
            The generated border patch tensor.
        """        
        if direction == "right":
            return generated[..., :, -patch_size:]
        elif direction == "left":
            return generated[..., :, :patch_size]
        elif direction == "down":
            return generated[..., -patch_size:, :]
        elif direction == "up":
            return generated[..., :patch_size, :]
        else:
            raise ValueError(f"Invalid direction {direction!r} for extracting new region.")

    def _stitch_image(
        self,
        accumulated: torch.Tensor,
        generated_patch: torch.Tensor,
        direction: str,
    ) -> torch.Tensor:
        """
        Stitch a newly generated patch onto the accumulated image.

        Args:
            accumulated: The image accumulated across prior iterations.
            generated_patch: Newly generated border patch.
            direction: Outpainting direction.

        Returns:
            The stitched image tensor.
        """
        if direction == "right":
            return torch.cat([accumulated, generated_patch], dim=-1)
        elif direction == "left":
            return torch.cat([generated_patch, accumulated], dim=-1)
        elif direction == "down":
            return torch.cat([accumulated, generated_patch], dim=-2)
        elif direction == "up":
            return torch.cat([generated_patch, accumulated], dim=-2)
        else:
            raise ValueError(f"Invalid direction {direction!r} for stitching image.")

    def _cyclic_shift(
        self,
        tensor: torch.Tensor,
        direction: str,
        patch_size: int,
    ) -> torch.Tensor:
        """
        Cyclically shift a tensor so the newly generated region becomes context
        for the next iteration.

        Args:
            tensor: Image or latent tensor to shift.
            direction: Outpainting direction.
            patch_size: Width or height of the generated border region.

        Returns:
            The cyclically shifted tensor.
        """
        if direction == "right":
            return torch.cat(
                [tensor[..., :, patch_size:], tensor[..., :, :patch_size]],
                dim=-1,
            )
        elif direction == "left":
            return torch.cat(
                [tensor[..., :, -patch_size:], tensor[..., :, :-patch_size]],
                dim=-1,
            )
        elif direction == "down":
            return torch.cat(
                [tensor[..., patch_size:, :], tensor[..., :patch_size, :]],
                dim=-2,
            )
        elif direction == "up":
            return torch.cat(
                [tensor[..., -patch_size:, :], tensor[..., :-patch_size, :]],
                dim=-2,
            )
        else:
            raise ValueError(f"Invalid direction {direction!r} for cyclic shift.")

    def _decode_latent_to_preview(
        self,
        latent: torch.Tensor,
        *,
        as_uint8: bool,
    ) -> np.ndarray:
        """
        Decode a latent tensor into an image preview.

        Args:
            latent: Latent tensor of shape ``(1, C, H_latent, W_latent)``.
            as_uint8: Whether to return the preview as ``uint8`` in ``[0, 255]``.
                If ``False``, returns floating-point values in ``[0, 1]``.

        Returns:
            A channel-last NumPy image array.
        """
        with torch.no_grad():
            decoded = self.vae.decode(latent / self.scaling_factor).sample  # type: ignore[attr-defined]

        preview = (
            decoded[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
        ).clip(0, 1)

        if as_uint8:
            return (preview * 255).astype(np.uint8)

        return preview

    def _save_iteration_outputs(
        self,
        *,
        latent: torch.Tensor,
        preview: np.ndarray,
        iteration: int,
        save_dir: str | Path,
        save_name: str,
    ) -> None:
        """
        Save the latent tensor and decoded preview for one outpainting iteration.

        Args:
            latent: Generated latent tensor for the iteration.
            preview: Decoded preview image as a NumPy array.
            iteration: Iteration index used in output filenames.
            save_dir: Root output directory.
            save_name: Per-run subdirectory name under ``save_dir``.
        """
        run_dir = Path(save_dir) / save_name
        run_dir.mkdir(parents=True, exist_ok=True)

        latent_path = run_dir / f"{iteration:02d}_latent.pt"
        image_path = run_dir / f"{iteration:02d}_image.png"

        torch.save(latent.detach().cpu(), latent_path)
        Image.fromarray(preview.astype(np.uint8)).save(image_path)

    def _run_one_from_config(self, cfg: OutpaintRunConfig) -> torch.Tensor:
        """
        Run outpainting inference for a single input image configuration.

        Args:
            cfg: Configuration describing the input image, output settings,
                denoising step count, crop ratio, direction, iteration count,
                and optional plotting behavior.

        Returns:
            The final generated latent tensor after all configured iterations.
        """
        current_image = self._load_image_tensor(cfg.input_dir)
        stitched = self._extract_old_region(
            current_image,
            direction=cfg.direction,
            crop_ratio=cfg.crop_ratio,
        )
        bbox = self._resolve_bbox(cfg)

        last_generated_latent: torch.Tensor | None = None
        current_latent: torch.Tensor | None = None

        for iteration in range(cfg.iterations):
            if iteration == 0:
                image_mask = self._create_image_mask(bbox, current_image.shape)
                masked_image = current_image * (1 - image_mask)

                with torch.no_grad():
                    masked_latent = self.vae.encode(masked_image).latent_dist.sample()  # type: ignore[attr-defined]
                    masked_latent = masked_latent * self.scaling_factor
            else:
                if current_latent is None:
                    raise RuntimeError(
                        "current_latent was unexpectedly None during outpaint iteration."
                    )
                masked_latent = current_latent

            latent_mask = self._create_latent_mask(bbox, masked_latent.shape)
            noise = torch.randn_like(masked_latent)

            noisy_region = self.noise_scheduler.add_noise(
                masked_latent * latent_mask,
                noise * latent_mask,
                torch.tensor(cfg.steps, device=self.device),  # type: ignore[arg-type]
            )
            noisy_latent = (
                masked_latent * (1 - latent_mask)
                + noisy_region * latent_mask
            )

            self.noise_scheduler.set_timesteps(cfg.num_steps, device=self.device)
            latent_input = noisy_latent

            bbox_encoder = self._require_bbox_encoder()
            cond_tokens = self.cond_encoder(masked_latent)
            seq_len = cond_tokens.shape[1]
            bbox_features = bbox_encoder(bbox).unsqueeze(1).expand(-1, seq_len, -1)
            condition = torch.cat([cond_tokens, bbox_features], dim=-1)

            for timestep in self.noise_scheduler.timesteps:
                latent_input = (
                    latent_input * latent_mask
                    + masked_latent * (1 - latent_mask)
                )
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_input,
                        timestep,
                        encoder_hidden_states=condition,
                    ).sample
                latent_input = self.noise_scheduler.step(
                    noise_pred,
                    timestep,  # type: ignore[arg-type]
                    latent_input,
                ).prev_sample  # type: ignore[attr-defined]

            generated_latent = (
                masked_latent * (1 - latent_mask)
                + latent_input * latent_mask
            )
            last_generated_latent = generated_latent

            with torch.no_grad():
                generated_image = self.vae.decode(
                    generated_latent / self.scaling_factor # type: ignore
                ).sample  # type: ignore[attr-defined]

            _, _, image_height, image_width = generated_image.shape
            _, _, latent_height, latent_width = generated_latent.shape

            if cfg.direction in {"right", "left"}:
                image_patch_size = image_width - int(image_width * cfg.crop_ratio)
                latent_patch_size = latent_width - int(latent_width * cfg.crop_ratio)
            else:
                image_patch_size = image_height - int(image_height * cfg.crop_ratio)
                latent_patch_size = latent_height - int(latent_height * cfg.crop_ratio)

            new_patch = self._extract_new_region(
                generated_image,
                direction=cfg.direction,
                patch_size=image_patch_size,
            )
            stitched = self._stitch_image(
                stitched,
                generated_patch=new_patch,
                direction=cfg.direction,
            )

            preview_uint8 = self._decode_latent_to_preview(
                generated_latent,
                as_uint8=True,
            )
            self._save_iteration_outputs(
                latent=generated_latent,
                preview=preview_uint8,
                iteration=iteration,
                save_dir=cfg.save_dir,
                save_name=cfg.save_name,
            )

            if cfg.show_plot:
                plot_decoded_image(
                    preview=preview_uint8,
                    iteration=iteration,
                    figsize=cfg.plot_fig_size,
                    title=cfg.plot_title,
                )

            current_image = self._cyclic_shift(
                generated_image,
                direction=cfg.direction,
                patch_size=image_patch_size,
            )
            current_latent = self._cyclic_shift(
                generated_latent,
                direction=cfg.direction,
                patch_size=latent_patch_size,
            )

        if last_generated_latent is None:
            raise RuntimeError("Outpaint inference produced no generated latent.")

        return last_generated_latent

    def run_one(
        self,
        image_path: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        num_steps: int = 200,
        crop_ratio: float = 0.97,
        iterations: int = 10,
        direction: str = "right",
        show_plot: bool = False,
        plot_title: str | None = "Outpainting Iteration",
        plot_fig_size: tuple[int, int] = (6, 6),
    ) -> torch.Tensor:
        """
        Run outpainting inference for a single image.

        This is the public single-image entry point. It builds an
        ``OutpaintRunConfig`` and delegates execution to the shared internal
        config-driven implementation.

        Args:
            input_dir: Path to the input image file.
            save_dir: Directory where outputs for this run should be saved.
            save_name: Name for the saved run subdirectory.
            num_steps: Number of denoising steps per iteration.
            crop_ratio: Fraction of the current image retained as context before
                generating the new border region.
            iterations: Number of iterative outpainting passes.
            direction: Direction in which to expand the image.
            show_plot: Whether to display intermediate and final previews.
            plot_title: Title for generated plots.
            plot_fig_size: Figure size for generated plots.

        Returns:
            The final generated latent tensor for the image.
        """
        cfg = OutpaintRunConfig(
            input_dir=image_path,
            save_dir=save_dir,
            save_name=save_name,
            num_steps=num_steps,
            crop_ratio=crop_ratio,
            iterations=iterations,
            direction=direction,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
        )
        return self._run_one_from_config(cfg)


    def run(
        self,
        input_dir: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        num_steps: int = 200,
        crop_ratio: float = 0.97,
        iterations: int = 10,
        direction: str = "right",
        show_plot: bool = False,
        plot_title: str = "Outpainting Iteration",
        plot_fig_size: tuple[int, int] = (6, 6),
    ) -> list[torch.Tensor]:
        """
        Run outpainting inference for every PNG image in a directory.

        Each file is processed through the same internal config-driven path used
        by ``run_one``. Output subdirectory names are derived from the base
        ``save_name`` and each file stem.

        Args:
            input_dir: Directory containing input PNG images.
            save_dir: Root directory where outputs should be saved.
            save_name: Base name used to build per-file run subdirectories.
            num_steps: Number of denoising steps per iteration.
            crop_ratio: Fraction of the current image retained as context before
                generating the new border region.
            iterations: Number of iterative outpainting passes per file.
            direction: Direction in which to expand each image.
            show_plot: Whether to display intermediate and final previews.
            plot_title: Title for generated plots.
            plot_fig_size: Figure size for generated plots.

        Returns:
            A list of final generated latent tensors, one per processed image.

        Raises:
            FileNotFoundError: If ``input_dir`` is not an existing directory.
        """
        cfg = OutpaintRunConfig(
            input_dir=input_dir,
            save_dir=save_dir,
            save_name=save_name,
            num_steps=num_steps,
            crop_ratio=crop_ratio,
            iterations=iterations,
            direction=direction,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
        )

        input_dir_path = Path(cfg.input_dir)
        if not input_dir_path.exists() or not input_dir_path.is_dir():
            raise FileNotFoundError(
                f"input_dir is not a directory: {input_dir_path}"
            )

        results: list[torch.Tensor] = []

        files = sorted(
            path
            for path in input_dir_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )

        for file_path in files:
            file_cfg = replace(
                cfg,
                input_dir=file_path,
                save_name=str(Path(cfg.save_name) / file_path.stem),
            )
            results.append(self._run_one_from_config(file_cfg))

        return results