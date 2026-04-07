from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler  # type: ignore

from .config import PixelDiffusionTrainerConfig
from .models import LatentAdapter, UNet512
from ...viz.cell_diagram_chart import save_side_by_side_barplot


type CascadeBatch = tuple[torch.Tensor, torch.Tensor]


class PixelDiffusionEvaluator:
    """
    Qualitative / analysis evaluator for stage-2 (512x512) cascade diffusion.

    Responsibilities:
    - generate qualitative samples from validation batches
    - save visualization grids
    - run composition evaluation using an auxiliary autoencoder
    - save side-by-side composition plots

    This class is intentionally separate from Cascade512Trainer so that
    training and evaluation/visualization concerns stay decoupled.
    """

    def __init__(
        self,
        *,
        adapter: LatentAdapter,
        unet512: UNet512,
        noise_scheduler: DDPMScheduler,
        val_loader,
        cfg: PixelDiffusionTrainerConfig,
        device: torch.device,
        accelerator: Accelerator | None = None,
        vae: AutoencoderKL | None = None,
    ) -> None:
        """
        Initialize evaluator runtime.

        Args:
            adapter: Conditional adapter used to compute multi-scale features.
            unet512: Stage-2 denoising UNet.
            noise_scheduler: Diffusion scheduler used for sampling.
            val_loader: Validation dataloader providing (target_img, z_cond).
            cfg: Shared trainer/evaluator config.
            device: Runtime device.
            accelerator: Optional accelerator used for logging/unwrap behavior.
            vae: Optional autoencoder used for composition evaluation.
        """
        self.adapter = adapter
        self.unet512 = unet512
        self.noise_scheduler = noise_scheduler
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.accelerator = accelerator
        self.vae = vae

        self.vis_dir = Path(self.cfg.vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _print(self, msg: str) -> None:
        """
        Print through Accelerator when available, otherwise fallback to print.
        """
        if self.accelerator is not None:
            self.accelerator.print(msg)
        else:
            print(msg)

    def _prepare_condition(self, z_cond: torch.Tensor) -> torch.Tensor:
        """
        Move conditional latent input onto the correct device/dtype.
        """
        return z_cond.to(self.device, dtype=torch.float16)

    def _set_eval_mode(self) -> None:
        """
        Put evaluation models into eval mode.
        """
        self.unet512.eval()
        self.adapter.eval()

        if self.vae is not None:
            self.vae.eval()

    @torch.no_grad()
    def _sample_batch(
        self,
        *,
        z_cond: torch.Tensor,
        target_shape: torch.Size,
        visualization_inference_steps: int,
    ) -> torch.Tensor:
        """
        Sample predicted images from a conditional latent batch.

        Args:
            z_cond: Conditional latent tensor.
            target_shape: Shape of target images, used to determine output size.

        Returns:
            Predicted image tensor in [-1, 1] space.
        """
        self._set_eval_mode()
        z_cond = self._prepare_condition(z_cond)

        # Compute multi-scale conditional features
        cond_feats = self.adapter(z_cond)
        batch_size = z_cond.size(0)

        self.noise_scheduler.set_timesteps(
            visualization_inference_steps,
            device=self.device,
        )

        # Start from pure noise
        x = (
            torch.randn(
                batch_size,
                3,
                target_shape[2],
                target_shape[3],
                device=self.device,
            )
            * self.noise_scheduler.init_noise_sigma
        )

        for timestep in self.noise_scheduler.timesteps:
            timestep_batch = torch.full(
                (batch_size,),
                int(timestep),
                device=self.device,
                dtype=torch.long,
            )
            x0_pred = self.unet512(x, timestep_batch, cond_feats)
            x = self.noise_scheduler.step(x0_pred, timestep, x).prev_sample  # type: ignore[assignment]

        return x

    def _infer_cell_map(
        self,
        img: torch.Tensor,
        ae_model: AutoencoderKL,
    ) -> torch.Tensor:
        """
        Runs autoencoder -> returns type map.
        """
        with torch.no_grad():
            return ae_model(img.unsqueeze(0))

    def _compute_type_distribution(
        self,
        type_map: np.ndarray | torch.Tensor,
        num_types: int = 25,
    ) -> np.ndarray:
        """
        Compute histogram of cell types.
        """
        if isinstance(type_map, torch.Tensor):
            type_map = type_map.detach().cpu().numpy()

        flat = type_map.flatten()
        hist, _ = np.histogram(flat, bins=num_types, range=(0, num_types))
        return hist / (hist.sum() + 1e-8)

    # Visualize sampled predictions
    @torch.no_grad()
    def visualize_epoch(
        self,
        epoch_idx: int,
        visualization_inference_steps: int,
        max_batches: int = 2,
    ) -> None:
        """
        Generate and save qualitative validation samples for a given epoch.
        """
        if epoch_idx < 0:
            raise ValueError(f"epoch_idx must be >= 0, got {epoch_idx!r}")

        self._set_eval_mode()

        saved = 0
        grids: list[torch.Tensor] = []

        for batch in self.val_loader:
            target_imgs, z_cond = batch
            batch_size = target_imgs.size(0)

            pred = self._sample_batch(
                z_cond=z_cond,
                target_shape=target_imgs.shape,
                visualization_inference_steps=visualization_inference_steps,
            )
            pred_vis = (pred.clamp(-1, 1) + 1) / 2
            target_vis = (target_imgs.clamp(-1, 1) + 1) / 2

            triplet = torch.cat([target_vis, pred_vis], dim=0)
            grid = vutils.make_grid(triplet, nrow=batch_size, padding=2)

            grids.append(grid)
            saved += 1
            if saved >= max_batches:
                break

        if grids:
            final_grid = torch.cat(grids, dim=1) if len(grids) > 1 else grids[0]
            out_path = self.vis_dir / f"epoch_{epoch_idx:03d}.png"
            vutils.save_image(final_grid, out_path)
            self._print(f"[Visualize] Saved {out_path}")

    # Composition Evaluation
    @torch.no_grad()
    def eval_composition_batch(
        self,
        *,
        epoch_idx: int,
        visualization_inference_steps: int,
        ae_model: AutoencoderKL | None = None,
        chart_left_title: str = "Original",
        chart_right_title: str = "Predicted",
    ) -> None:
        """
        Generate one validation batch, infer predicted/original type
        distributions, and save side-by-side composition plots.

        Args:
            epoch_idx: Epoch index used in output file naming.
            visualization_inference_steps: Number of inference steps for visualization.
            ae_model: Optional override autoencoder for type-map inference.
                Falls back to self.vae if not provided.
        """
        self._set_eval_mode()

        ae = ae_model if ae_model is not None else self.vae
        if ae is None:
            raise RuntimeError("VAE model must be provided for composition evaluation.")

        batch = next(iter(self.val_loader))
        target_imgs, z_cond = batch
        batch_size = target_imgs.size(0)

        pred_imgs = self._sample_batch(
            z_cond=z_cond,
            target_shape=target_imgs.shape,
            visualization_inference_steps=visualization_inference_steps,
        )
        pred_vis = (pred_imgs.clamp(-1, 1) + 1) / 2
        target_vis = (target_imgs.clamp(-1, 1) + 1) / 2

        # Compute compositions
        predicted_fractions: list[np.ndarray] = []
        original_fractions: list[np.ndarray] = []

        for i in range(batch_size):
            type_pred = self._infer_cell_map(pred_vis[i], ae)
            type_orig = self._infer_cell_map(target_vis[i], ae)

            dist_pred = self._compute_type_distribution(
                type_pred.squeeze().cpu().numpy()
            )
            dist_orig = self._compute_type_distribution(
                type_orig.squeeze().cpu().numpy()
            )

            predicted_fractions.append(dist_pred)
            original_fractions.append(dist_orig)

        # Save plots
        xs = np.arange(len(predicted_fractions[0]))

        for i in range(batch_size):
            out_path = self.vis_dir / f"comp_eval_{epoch_idx}_img{i}.png"
            save_side_by_side_barplot(
                xs=xs,
                left_vals=np.asarray(original_fractions[i]),
                right_vals=np.asarray(predicted_fractions[i]),
                left_title=chart_left_title,
                right_title=chart_right_title,
                out_path=out_path,
            )
            self._print(f"[CompEval] Saved {out_path}")
