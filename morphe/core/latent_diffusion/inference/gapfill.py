from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Self, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel # type: ignore

from ..artifact import LatentDiffusionArtifact
from .base import BaseLatentInferencer
from .run_config import GapfillRunConfig
from ..model import BBoxEncoder, CondEncoder
from ....constants import InferenceMode
from ....utils import resolve_device, resolve_dtype
from ....viz.decoded_img import plot_decoded_image


class GapfillInferencer(BaseLatentInferencer):
    """
    Inference wrapper for latent-space gap-filling generation.

    This inferencer reconstructs gap-filled latent images by repeatedly masking
    a target horizontal band, denoising the masked region with diffusion, and
    stitching the result back into the latent for iterative expansion.
    """

    def __init__(
        self,
        *,
        artifact: LatentDiffusionArtifact,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        cond_encoder: CondEncoder,
        coord_encoder: None = None,
        bbox_encoder: BBoxEncoder,
        noise_scheduler: DDPMScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize a gap-filling inferencer from fully constructed runtime components.

        This validates that the artifact was trained for gap-filling inference
        and that all required runtime components are present before delegating
        shared setup to ``BaseLatentInferencer``.

        Args:
            artifact: Trained latent diffusion artifact for gap-filling.
            unet: Reconstructed diffusion UNet.
            vae: Reconstructed variational autoencoder.
            cond_encoder: Conditioning encoder used during inference.
            coord_encoder: Optional coordinate encoder.
            bbox_encoder: Bounding-box encoder required for gap-filling.
            noise_scheduler: Diffusion scheduler used for denoising.
            device: Device to run inference on.
            dtype: Floating-point dtype to use for inference.

        Raises:
            RuntimeError: If the artifact was not created for gap-filling
                inference, if the bbox encoder is missing, or if
                ``artifact.img_size`` is not defined.
        """
        if artifact.inference_mode != InferenceMode.GAPFILL:
            raise RuntimeError(
                "GapfillInferencer requires artifact.inference_mode == "
                f"{InferenceMode.GAPFILL!r}, got {artifact.inference_mode!r}."
            )

        if bbox_encoder is None or not isinstance(bbox_encoder, BBoxEncoder):
            raise RuntimeError(
                "GapfillInferencer requires a bbox encoder in the artifact runtime."
            )

        if artifact.img_size is None:
            raise RuntimeError(
                "GapfillInferencer requires artifact.img_size to be defined."
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

        self.image_transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
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
        Construct a gap-filling inferencer from a latent diffusion artifact.

        This method reconstructs all required runtime components from the
        artifact, validates that the artifact is compatible with gap-filling
        inference, and initializes the inferencer instance.

        Args:
            artifact: Latent diffusion artifact containing trained weights and
                configuration for gap-filling inference.
            device: Target device for runtime inference components.
            dtype: Target floating-point dtype for inference components. When
                ``None``, defaults to a suitable dtype based on the device.
        """
        resolved_device = resolve_device(device)
        resolved_dtype = torch.float32

        unet, vae, cond_encoder, _, bbox_encoder, noise_scheduler = (
            artifact.architecture.build_components(
                device=resolved_device,
                dtype=resolved_dtype,
            )
        )
        
        if cond_encoder is None or not isinstance(cond_encoder, CondEncoder):
            raise RuntimeError(
                "GapfillInferencer requires a cond encoder in the artifact architecture."
            )
            
        if bbox_encoder is None or not isinstance(bbox_encoder, BBoxEncoder):
            raise RuntimeError(
                "GapfillInferencer requires a bbox encoder in the artifact architecture."
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
 

    def _create_latent_mask(
        self,
        bbox: torch.Tensor,
        latent_shape: Sequence[int],
    ) -> torch.Tensor:
        """
        Build a binary latent-space mask from normalized bounding boxes.

        The returned mask has value ``1`` inside each bbox region and ``0``
        elsewhere, in latent-space coordinates.

        Args:
            bbox: Tensor of normalized bounding boxes with shape ``(N, 4)``,
                where each row is ``[x1, y1, x2, y2]`` in ``[0, 1]``.
            latent_shape: Shape of the latent tensor, typically
                ``(B, C, H_latent, W_latent)``.

        Returns:
            A mask tensor of shape ``(N, 1, H_latent, W_latent)`` on the
            inferencer device.
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

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        """
        Load, normalize, and batch an input image for latent inference.

        The image is converted to RGB, resized to the artifact's configured
        spatial size, normalized to ``[-1, 1]``, and moved onto the inferencer
        device.

        Args:
            image_path: Path to the input image file.

        Returns:
            A tensor of shape ``(1, 3, H, W)`` on the inferencer device.
        """
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = self.image_transform(rgb_image).unsqueeze(0)

        return image_tensor.to(self.device)

    def _resolve_bbox(self, cfg: GapfillRunConfig) -> torch.Tensor:
        """
        Resolve the bounding box tensor for a single run.

        Falls back to the historical default central horizontal band if no bbox
        is provided in the config.

        Args:
            cfg: Gap-filling run configuration.

        Returns:
            A bbox tensor of shape ``(N, 4)`` on the inferencer device.
        """
        if cfg.bbox is not None:
            return cfg.bbox

        return torch.tensor(
            [[0.0, 0.4375, 1.0, 0.5625]],
            device=self.device,
        )

    def _encode_image_to_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode an image tensor into latent space.

        The encoded latent is sampled from the VAE posterior and scaled by the
        inferencer's latent scaling factor.

        Args:
            image_tensor: Normalized image tensor of shape ``(1, 3, H, W)``.

        Returns:
            A latent tensor of shape ``(1, C, H_latent, W_latent)``.
        """
        with torch.no_grad():
            latent = self.encode_with_vae(image_tensor)

        return latent

    def _make_condition(
        self,
        masked_latent: torch.Tensor,
        bbox: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the conditioning tensor for gap-filling denoising.

        This combines latent-derived conditioning tokens from the conditioning
        encoder with bbox embedding features from the bbox encoder.

        Args:
            masked_latent: Masked latent tensor of shape
                ``(B, C, H_latent, W_latent)``.
            bbox: Bounding box tensor of shape ``(B, 4)`` or compatible shape
                expected by the bbox encoder.

        Returns:
            A conditioning tensor of shape ``(B, seq_len, D_total)`` suitable
            for passing as ``encoder_hidden_states`` to the UNet.

        Raises:
            RuntimeError: If the bbox encoder is not available.
        """
        bbox_encoder = self.bbox_encoder
        if bbox_encoder is None:
            raise RuntimeError(
                "GapfillInferencer requires a bbox encoder in the artifact runtime."
            )

        seq_len = self.cond_encoder(masked_latent).shape[1]
        bbox_features = bbox_encoder(bbox).unsqueeze(1).expand(-1, seq_len, -1)

        return torch.cat(
            [
                self.cond_encoder(masked_latent),
                bbox_features,
            ],
            dim=-1,
        )

    def _decode_latent_to_preview(
        self,
        latent: torch.Tensor,
        *,
        as_uint8: bool,
    ) -> np.ndarray:
        """
        Decode a latent tensor into an image preview for visualization or saving.

        Args:
            latent: Latent tensor of shape ``(1, C, H_latent, W_latent)``.
            as_uint8: Whether to return the preview as ``uint8`` in ``[0, 255]``.
                If ``False``, returns floating-point values in ``[0, 1]``.

        Returns:
            A channel-last NumPy image array suitable for plotting or saving.
        """
        for param in self.vae.parameters():
            param.data = param.data.to(dtype=torch.float32)
        for buf in self.vae.buffers():
            buf.data = buf.data.to(dtype=torch.float32)

        with torch.no_grad():
            decoded = self.vae.decode((latent / self.scaling_factor).to(dtype=torch.float32)).sample  # type: ignore

        preview = (
            decoded[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        ).clip(0, 1)

        if as_uint8:
            return (preview * 255).astype(np.uint8)

        return preview

    def _save_final_outputs(
        self,
        *,
        latent: torch.Tensor,
        preview: np.ndarray,
        iteration: int,
        save_dir: str | Path,
        save_name: str,
    ) -> None:
        """
        Save the final latent tensor and decoded preview image for a run.

        Outputs are written into a per-run directory under ``save_dir``, with
        filenames derived from the iteration index.

        Args:
            latent: Final generated latent tensor to serialize.
            preview: Decoded image preview to save as a PNG.
            iteration: Iteration index used in output filenames.
            save_dir: Root directory for saved outputs.
            save_name: Per-run subdirectory name under ``save_dir``.
        """
        if not save_dir:
            return
        
        save_dir = Path(save_dir)
        run_dir = save_dir / save_name
        run_dir.mkdir(parents=True, exist_ok=True)

        latent_path = run_dir / f"{iteration}_latent.pt"
        image_path = run_dir / f"{iteration}_image.png"

        torch.save(latent.detach().cpu(), latent_path)
        Image.fromarray(preview.astype(np.uint8)).save(image_path)

    def _get_gap_columns(
        self,
        bbox: torch.Tensor,
        latent_width: int,
    ) -> tuple[int, int]:
        """
        Convert the first bbox's x-range into latent-space column indices.

        Args:
            bbox: Bounding box tensor of shape ``(N, 4)`` in normalized
                coordinates.
            latent_width: Width of the latent tensor in pixels.

        Returns:
            A half-open interval ``[start, end)`` in latent-width coordinates.

        Raises:
            ValueError: If ``bbox`` does not have shape ``(N, 4)`` or if the
                resolved interval is empty or invalid.
        """
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise ValueError(
                f"Expected bbox with shape (N, 4), got {tuple(bbox.shape)}."
            )

        x1 = float(bbox[0, 0].item())
        x2 = float(bbox[0, 2].item())

        start = int(round(x1 * latent_width))
        end = int(round(x2 * latent_width))

        start = max(0, min(start, latent_width))
        end = max(0, min(end, latent_width))

        if end <= start:
            raise ValueError(
                f"Resolved invalid latent gap interval [{start}, {end}) from bbox {bbox[0]}."
            )

        return start, end

    def _stitch_next_latent(
        self,
        current_latent: torch.Tensor,
        latent_input: torch.Tensor,
        bbox: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct the next latent tensor for the following gap-filling iteration.

        This method extracts generated side patches from the current denoised
        latent, preserves the appropriate outer context, inserts a fresh empty
        gap region, and reassembles a latent of the original width.

        Args:
            current_latent: Current latent tensor before stitching.
            latent_input: Latest denoised latent tensor containing generated
                content around the gap region.
            bbox: Bounding box tensor used to determine the gap geometry.

        Returns:
            A stitched latent tensor with the same shape as ``current_latent``.

        Raises:
            ValueError: If the resolved geometry is invalid or too narrow to
                construct a valid stitched latent.
        """
        latent_width = current_latent.shape[-1]
        gap_start, gap_end = self._get_gap_columns(bbox, latent_width)
        gap_width = gap_end - gap_start

        if gap_width < 2:
            raise ValueError(
                f"Gap width must be at least 2 latent pixels, got {gap_width}."
            )

        side_patch_width = gap_width // 2
        if side_patch_width == 0:
            raise ValueError(
                f"Derived side patch width is zero for gap width {gap_width}."
            )

        center_patch = latent_input[..., gap_start:gap_end]
        left_patch = center_patch[..., :side_patch_width]
        right_patch = center_patch[..., gap_width - side_patch_width:]

        cropped_latent = current_latent[..., side_patch_width:-side_patch_width]
        cropped_width = cropped_latent.shape[-1]

        remaining_width = latent_width - (2 * side_patch_width + gap_width)
        if remaining_width < 0:
            raise ValueError(
                "Resolved stitch geometry is invalid: remaining context width is negative."
            )

        left_context_width = remaining_width // 2
        right_context_width = remaining_width - left_context_width

        cropped_left = cropped_latent[..., :left_context_width]
        cropped_right = cropped_latent[..., cropped_width - right_context_width:]

        white_gap_latent = torch.zeros(
            (
                current_latent.shape[0],
                current_latent.shape[1],
                current_latent.shape[2],
                gap_width,
            ),
            device=self.device,
            dtype=current_latent.dtype,
        )

        stitched_latent = torch.cat(
            [
                cropped_left,
                left_patch,
                white_gap_latent,
                right_patch,
                cropped_right,
            ],
            dim=-1,
        )

        if stitched_latent.shape[-1] != latent_width:
            start = (stitched_latent.shape[-1] - latent_width) // 2
            stitched_latent = stitched_latent[..., start:start + latent_width]

        return stitched_latent
    
    def _run_one_from_config(self, cfg: GapfillRunConfig) -> torch.Tensor:
        """
        Run gap-filling inference for a single input image configuration.

        This method loads the input image, encodes it into latent space, applies
        iterative masked denoising over the configured gap region, optionally
        visualizes intermediate results, and returns the final generated latent.

        Args:
            cfg: Configuration describing the input image, output settings,
                denoising step count, iteration count, plotting options, and
                optional bbox override.

        Returns:
            The final generated latent tensor after all configured iterations.
        """
        unet_dtype = next(self.unet.parameters()).dtype

        image_tensor = self._load_image_tensor(Path(cfg.input_dir))
        current_latent = self._encode_image_to_latent(image_tensor).to(dtype=unet_dtype)
        bbox = self._resolve_bbox(cfg).to(dtype=unet_dtype)

        for iteration in range(cfg.iterations):
            latent_mask = self._create_latent_mask(bbox, current_latent.shape).to(dtype=unet_dtype)
            masked_latent = current_latent * (1 - latent_mask)

            noise = torch.randn_like(current_latent)
            noisy_latent_region = self.noise_scheduler.add_noise(
                masked_latent * latent_mask,
                noise * latent_mask,
                torch.tensor(cfg.num_steps),  # type: ignore
            )
            noisy_latent = (
                masked_latent * (1 - latent_mask)
                + noisy_latent_region * latent_mask
            )

            self.noise_scheduler.set_timesteps(cfg.num_steps)
            latent_input = noisy_latent
            condition = self._make_condition(masked_latent, bbox=bbox)

            for step_idx, timestep in enumerate(self.noise_scheduler.timesteps, start=1):
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
                    timestep, # type: ignore
                    latent_input,
                ).prev_sample.to(dtype=unet_dtype)  # type: ignore

                if step_idx == cfg.num_steps - 1:
                    generated_latent = (
                        latent_input * latent_mask
                        + masked_latent * (1 - latent_mask)
                    )
                    preview_uint8 = self._decode_latent_to_preview(
                        generated_latent,
                        as_uint8=True,
                    )

            if iteration == cfg.iterations - 1:
                generated_latent = (
                    latent_input * latent_mask
                    + masked_latent * (1 - latent_mask)
                )
                preview_uint8 = self._decode_latent_to_preview(
                    generated_latent,
                    as_uint8=True,
                )

                self._save_final_outputs(
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
                    
                return generated_latent

            current_latent = self._stitch_next_latent(
                current_latent=current_latent,
                latent_input=latent_input,
                bbox=bbox,
            )

            preview_float = self._decode_latent_to_preview(
                current_latent,
                as_uint8=False,
            )
            if cfg.show_plot:
                plot_decoded_image(
                    preview=preview_float,
                    iteration=iteration,
                    figsize=cfg.plot_fig_size,
                    title=cfg.plot_title,
                )

        return current_latent
        

    def run_one(
        self,
        image_path: str | Path,
        save_dir: str | Path,
        run_name: str = "default",
        num_steps: int = 200,
        num_iterations: int = 10,
        show_plot: bool = False,
        plot_title: str | None = None,
        plot_fig_size: tuple[int, int] = (6, 6),
        gap_bbox: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Run gap-filling inference for a single image.

        This is the public single-image entry point. It builds a
        ``GapfillRunConfig`` and delegates execution to the shared internal
        config-driven implementation.

        Args:
            image_path: Path to the input image file.
            save_dir: Directory where outputs for this run should be saved.
            run_name: Name for the saved run subdirectory.
            num_steps: Number of denoising steps per iteration.
            iterations: Number of iterative gap-filling passes.
            show_plot: Whether to display intermediate and final previews.
            plot_title: Optional title for generated plots.
            plot_fig_size: Figure size for generated plots.
            gap_bbox: Optional normalized bbox override for the gap region.

        Returns:
            The final generated latent tensor for the image.
        """
        cfg = GapfillRunConfig(
            input_dir=image_path,
            save_dir=save_dir,
            save_name=run_name,
            num_steps=num_steps,
            iterations=num_iterations,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
            bbox=gap_bbox,
        )
        
        return self._run_one_from_config(cfg)

    def run(
        self,
        input_dir: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        num_steps: int = 200,
        iterations: int = 10,
        show_plot: bool = False,
        plot_title: str | None = None,
        plot_fig_size: tuple[int, int] = (6, 6),
        bbox: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Run gap-filling inference for every PNG image in a directory.

        Each file is processed through the same internal config-driven path used
        by ``run_one``. Output subdirectory names are derived from the base
        ``save_name`` and each file stem.

        Args:
            input_dir: Directory containing input PNG images.
            save_dir: Root directory where outputs should be saved.
            save_name: Base name used to build per-file run subdirectories.
            num_steps: Number of denoising steps per iteration.
            iterations: Number of iterative gap-filling passes per file.
            show_plot: Whether to display intermediate and final previews.
            plot_title: Optional title for generated plots.
            plot_fig_size: Figure size for generated plots.
            bbox: Optional normalized bbox override applied to every file.

        Returns:
            A list of final generated latent tensors, one per processed image.
        """
        cfg = GapfillRunConfig(
            input_dir=input_dir,
            save_dir=save_dir,
            save_name=save_name,
            num_steps=num_steps,
            iterations=iterations,
            show_plot=show_plot,
            plot_title=plot_title,
            plot_fig_size=plot_fig_size,
            bbox=bbox,
        )
        results: list[torch.Tensor] = []

        files = sorted(
            path
            for path in cfg.input_dir.iterdir() # type: ignore
            if path.is_file() and path.suffix.lower() == ".png"
        )

        for file_path in files:
            file_cfg = replace(
                cfg,
                input_dir=file_path,
                save_name=str(Path(cfg.save_name) / file_path.stem),
            )
            result_latent = self._run_one_from_config(file_cfg)
            results.append(result_latent)

        return results