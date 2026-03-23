from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel # type: ignore

from disco.config import InferenceMode
from disco.core.latent_diffusion.infer.base import (
    BaseLatentInferencer
)
from disco.core.latent_diffusion.infer.run_config import InpaintRunConfig
from disco.core.latent_diffusion.model import BBoxEncoder, CondEncoder, CoordEncoder
from disco.core.latent_diffusion.artifact import LatentDiffusionArtifact
from disco.viz.decoded_img import plot_decoded_image, plot_inpainting_triplet


class InpaintInferencer(BaseLatentInferencer):
    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        cond_encoder: CondEncoder,
        coord_encoder: CoordEncoder,
        bbox_encoder: None,
        noise_scheduler: DDPMScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize an inferencer for latent diffusion inpainting.

        This inferencer validates that the artifact was trained for inpainting
        and that all required inpainting-specific components are present before
        delegating shared runtime setup to ``BaseLatentInferencer``.

        Args:
            artifact: Trained latent diffusion artifact for inpainting.
            device: Device to run inference on. If ``None``, a default device
                is resolved by the base inferencer.
            dtype: Floating-point dtype to use for inference. If ``None``,
                a default dtype is resolved by the base inferencer.

        Raises:
            RuntimeError: If the artifact was not created for inpainting, if
                the coordinate encoder is missing, or if ``artifact.img_size``
                is not defined.
        """
        if artifact.inference_mode != InferenceMode.INPAINTING:
            raise RuntimeError(
                f"InpaintInferencer requires an artifact with "
                f"inference_mode={InferenceMode.INPAINTING}, "
                f"but got {artifact.inference_mode}."
            )

        if coord_encoder is None:
            raise RuntimeError(
                "InpaintInferencer requires coord_encoder to be present in the artifact."
            )

        if artifact.img_size is None:
            raise RuntimeError(
                "InpaintInferencer requires artifact.img_size to be defined."
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

    def _load_image(self, path: str | Path) -> torch.Tensor:
        """
        Load and normalize an input image for latent inpainting.

        The image is resized to the artifact's configured spatial size,
        converted to a tensor in ``CHW`` format, normalized to ``[-1, 1]``,
        batched, and moved onto the inferencer device and dtype.

        Args:
            path: Path to the input image file.

        Returns:
            A tensor of shape ``(1, 3, H, W)`` on the inferencer device.
        """
        img = Image.open(path).convert("RGB")
        img = img.resize((self.artifact.img_size, self.artifact.img_size))  # type: ignore[arg-type]
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        img_tensor = img_tensor * 2 - 1  # [-1, 1]
        return img_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _load_mask(self, path: str | Path) -> torch.Tensor:
        """
        Load and normalize an inpainting mask.

        The mask is resized to the artifact's configured spatial size,
        converted to a single-channel tensor, batched, and moved onto the
        inferencer device and dtype. White regions are treated as masked
        regions and black regions as preserved regions.

        Args:
            path: Path to the mask image file.

        Returns:
            A tensor of shape ``(1, 1, H, W)`` on the inferencer device, where
            masked pixels are near ``1`` and preserved pixels are near ``0``.
        """
        mask = (
            Image.open(path)
            .convert("L")
            .resize((self.artifact.img_size, self.artifact.img_size))  # type: ignore[arg-type]
        )
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return mask_tensor.to(device=self.device, dtype=self.dtype)

    def _require_coord_encoder(self) -> CoordEncoder:
        """
        Return the coordinate encoder required for inpainting inference.

        This helper provides a narrow, explicit check when coordinate-based
        conditioning is needed during execution.

        Returns:
            The coordinate encoder stored on the inferencer.

        Raises:
            RuntimeError: If the coordinate encoder is not available.
        """
        if self.coord_encoder is None:
            raise RuntimeError("coord_encoder is required for inpaint inference.")
        return self.coord_encoder

    def _to_numpy_image(self, x: torch.Tensor) -> np.ndarray:
        """
        Convert a tensor image-like value into a NumPy array for plotting.

        This helper removes a leading batch dimension when present and converts
        channel-first tensors into channel-last layout for visualization.

        Args:
            x: Tensor containing an image or mask. Supported shapes include
                ``(1, C, H, W)``, ``(C, H, W)``, or already spatial arrays.

        Returns:
            A NumPy array suitable for visualization.
        """
        x = x.detach().cpu()

        if x.ndim == 4:
            x = x[0]

        if x.ndim == 3 and x.shape[0] in {1, 3}:
            x = x.permute(1, 2, 0)

        return x.numpy()

    def _plot_masked_preview(
        self,
        *,
        masked_latent: torch.Tensor,
        cfg: InpaintRunConfig,
    ) -> None:
        """
        Decode and display the masked latent preview before denoising.

        This visualization is useful for confirming that the masked region and
        preserved context were constructed correctly before the diffusion loop.

        Args:
            masked_latent: Partially masked latent tensor of shape
                ``(1, C, H_latent, W_latent)``.
            cfg: Inference configuration controlling plot title and figure size.
        """
        with torch.no_grad():
            preview = self.vae.decode(masked_latent / self.scaling_factor).sample  # type: ignore[attr-defined]

        preview = (preview.clamp(-1, 1) + 1) / 2
        preview_np = preview[0].permute(1, 2, 0).detach().cpu().numpy()

        plot_decoded_image(
            preview=preview_np,
            figsize=cfg.plot_fig_size,
            title=cfg.plot_title,
        )

    def _plot_final_triplet(
        self,
        *,
        image: torch.Tensor,
        mask: torch.Tensor,
        result: torch.Tensor,
        cfg: InpaintRunConfig,
    ) -> None:
        """
        Display the original image, mask, and reconstructed inpainting result.

        This helper converts tensors into visualization-friendly arrays and
        renders a side-by-side comparison of the input image, the inpainting
        mask, and the final decoded output.

        Args:
            image: Original normalized input image tensor of shape
                ``(1, 3, H, W)``.
            mask: Inpainting mask tensor of shape ``(1, 1, H, W)``.
            result: Decoded reconstructed image tensor of shape
                ``(1, 3, H, W)``.
            cfg: Inference configuration controlling figure size.
        """
        result_disp = (result.clamp(-1, 1) + 1) / 2
        result_np = result_disp[0].permute(1, 2, 0).detach().cpu().numpy()

        plot_inpainting_triplet(
            image=self._to_numpy_image((image.clamp(-1, 1) + 1) / 2),
            mask=self._to_numpy_image(mask),
            result=result_np,
            figsize=cfg.plot_fig_size,
        )

    def _run_one_from_config(self, cfg: InpaintRunConfig) -> torch.Tensor:
        """
        Run latent diffusion inpainting for a single image/mask pair.

        This method performs the full inpainting pipeline:
        loading inputs, encoding the image into latent space, constructing the
        masked latent and latent-space mask, preparing conditioning features,
        iteratively denoising the masked region, and optionally visualizing
        intermediate and final outputs.

        Args:
            cfg: Configuration describing the image path, mask path, denoising
                step count, and optional plotting behavior.

        Returns:
            The final inpainted latent tensor of shape
            ``(1, C, H_latent, W_latent)``.
        """
        image = self._load_image(cfg.input_dir)
        mask = self._load_mask(cfg.mask_dir)

        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample() * self.scaling_factor  # type: ignore[attr-defined]

        _, channels, latent_h, latent_w = latent.shape

        latent_mask = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
        latent_mask = latent_mask.expand(-1, channels, -1, -1)

        masked_latent = latent * (1 - latent_mask)

        if cfg.show_plot:
            self._plot_masked_preview(masked_latent=masked_latent, cfg=cfg)

        self.noise_scheduler.set_timesteps(cfg.num_steps, device=self.device)

        noisy = torch.randn_like(latent)
        x = masked_latent + noisy * latent_mask

        cond_tokens = self.cond_encoder(masked_latent)
        coord_tokens = self._require_coord_encoder()(mask)
        condition = torch.cat([cond_tokens, coord_tokens], dim=-1)

        for t in self.noise_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.unet(x, t, encoder_hidden_states=condition).sample

            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample  # type: ignore[attr-defined]
            x = latent_mask * x + (1 - latent_mask) * masked_latent

        if cfg.show_plot:
            with torch.no_grad():
                image_recon = self.vae.decode(x / self.scaling_factor).sample  # type: ignore[attr-defined]
            self._plot_final_triplet(
                image=image,
                mask=mask,
                result=image_recon,
                cfg=cfg,
            )

        return x

    def run_one(
        self,
        image_path: str | Path,
        mask_dir: str | Path,
        num_steps: int = 200,
        show_plot: bool = False,
        plot_title: str = "Inpainting Progress",
        plot_fig_size: tuple[int, int] = (6, 6),
    ) -> torch.Tensor:
        """
        Run inpainting for a single image/mask pair.

        Args:
            input_dir: Path to the input image file.
            mask_dir: Path to the mask image file.
            num_steps: Number of diffusion inference steps.
            show_plot: Whether to display a comparison plot.
            plot_title: Title for the optional plot.
            plot_fig_size: Figure size for the optional plot.

        Returns:
            The final inpainted image tensor.
        """
        cfg = InpaintRunConfig(
            input_dir=image_path,
            mask_dir=mask_dir,
            num_steps=num_steps,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
        )
        return self._run_one_from_config(cfg)


    def run(
        self,
        input_dir: str | Path,
        mask_dir: str | Path,
        num_steps: int = 200,
        show_plot: bool = False,
        plot_title: str = "Inpainting Progress",
        plot_fig_size: tuple[int, int] = (6, 6),
    ) -> list[torch.Tensor]:
        """
        Run inpainting over every valid image in an input directory.

        Each image in ``input_dir`` is matched with a mask of the same filename
        in ``mask_dir``. Files without a corresponding mask are skipped.

        Args:
            input_dir: Directory containing input image files.
            mask_dir: Directory containing mask image files.
            num_steps: Number of diffusion inference steps per sample.
            show_plot: Whether to display a comparison plot for each sample.
            plot_title: Title for optional plots.
            plot_fig_size: Figure size for optional plots.

        Returns:
            A list of inpainted image tensors, one per successfully processed file.

        Raises:
            FileNotFoundError: If ``input_dir`` or ``mask_dir`` is not a directory.
        """
        input_dir = Path(input_dir)
        mask_dir = Path(mask_dir)

        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"input_dir is not a directory: {input_dir}")

        if not mask_dir.exists() or not mask_dir.is_dir():
            raise FileNotFoundError(f"mask_dir is not a directory: {mask_dir}")

        results: list[torch.Tensor] = []

        image_files = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        for image_path in image_files:
            per_file_mask_path = mask_dir / image_path.name

            if not per_file_mask_path.exists():
                warnings.warn(
                    f"No mask found for {image_path.name}; skipping.",
                    stacklevel=2,
                )
                continue

            per_file_cfg = InpaintRunConfig(
                input_dir=image_path,
                mask_dir=per_file_mask_path,
                num_steps=num_steps,
                show_plot=show_plot,
                plot_title=plot_title,
                plot_fig_size=plot_fig_size,
            )
            results.append(self._run_one_from_config(per_file_cfg))

        return results