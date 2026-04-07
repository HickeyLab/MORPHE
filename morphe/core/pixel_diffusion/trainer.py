from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm


from .dataset import PrecomputedCascadeDataset
from .config import PixelDiffusionTrainerConfig
from .artifact import PixelDiffusionArtifact
from .evaluator import PixelDiffusionEvaluator
from .models import LatentAdapter, UNet512
from ...utils import get_config_attr, resolve_device, set_config_attr


class PixelDiffusionTrainer:
    """
    Stage-2 (512×512) cascade diffusion trainer.

    Responsibilities:
    - load precomputed cascade datasets
    - train adapter + UNet on pixel-space denoising
    - run validation
    - save / restore best checkpoint
    - return a serializable PixelDiffusionArtifact
    """

    def __init__(
        self,
        *,
        train_index: str | Path,
        val_index: str | Path,
        cfg: PixelDiffusionTrainerConfig,
        device: torch.device | str | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize trainer state and construct the full training runtime.

        This includes:
        - path normalization + validation
        - accelerator creation
        - dataloader construction
        - model construction
        - optimizer construction
        - scheduler construction
        - accelerator preparation
        - evaluator construction

        Args:
            train_index: Path to the training index file.
            val_index: Path to the validation index file.
            cfg: Trainer configuration.
            device: Optional requested runtime device. When set to CPU, the
                accelerator is forced onto CPU. When set to CUDA, training will
                require CUDA to be available.
        """
        self.cfg = cfg
        self.global_step: int = 0
        self.seed = seed
        self.verbose = verbose

        self.train_index = Path(train_index)
        self.val_index = Path(val_index)
        self._validate_indices()

        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self.requested_device = resolve_device(device)

        self._set_seed()

        # AMP + distributed manager
        self.accelerator = self._build_accelerator()
        self.device = self.accelerator.device

        if self.requested_device.type == "cuda" and self.device.type != "cuda":
            raise ValueError(
                f"Requested CUDA device {self.requested_device}, "
                f"but accelerator resolved to {self.device}."
            )

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.adapter, self.unet512, self.vae = self._build_models()
        self.optimizer = self._build_optimizer()
        self.noise_scheduler = self._build_noise_scheduler()

        # Prepare all for Accelerator
        self._prepare_runtime()

        # Evaluator
        self.evaluator = PixelDiffusionEvaluator(
            adapter=self.adapter,
            unet512=self.unet512,
            noise_scheduler=self.noise_scheduler,
            val_loader=self.val_loader,
            cfg=self.cfg,
            device=self.device,
            accelerator=self.accelerator,
            vae=self.vae,
            verbose=self.verbose,
        )

        if self.verbose:
            self.accelerator.print(
                "Initialized PixelDiffusionTrainer "
                f"(device={self.device}, seed={self.seed}, "
                f"train_batches={len(self.train_loader)}, "
                f"val_batches={len(self.val_loader)})"
            )

    def _set_seed(self) -> None:
        """
        Seed Python, NumPy, and PyTorch RNGs when a seed is provided.
        """
        if self.seed is None:
            return

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        if self.verbose:
            print(f"Set random seed to {self.seed}.")

    def _build_accelerator(self) -> Accelerator:
        """
        Build the Accelerate runtime manager.

        Returns:
            Configured accelerator instance.

        Notes:
            CPU is forced only when the requested device is CPU. Otherwise,
            Accelerate resolves the appropriate runtime device.
        """
        return Accelerator(
            mixed_precision="fp16",
            cpu=self.requested_device.type == "cpu",
        )

    def _build_dataloaders(
        self,
    ) -> tuple[
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Construct train and validation dataloaders from the precomputed indices.
        """
        train_loader = DataLoader(
            PrecomputedCascadeDataset(self.train_index),
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.train_num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            PrecomputedCascadeDataset(self.val_index),
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.val_num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def _build_models(
        self,
    ) -> tuple[LatentAdapter, UNet512, AutoencoderKL | None]:
        """
        Construct model components used by the trainer.
        """
        adapter_kwargs = self.cfg.adapter_kwargs or {}
        unet_kwargs = self.cfg.unet_kwargs or {}

        adapter = LatentAdapter(**adapter_kwargs)  # type: ignore[arg-type]
        unet512 = UNet512(**unet_kwargs)

        vae: AutoencoderKL | None = AutoencoderKL.from_pretrained(
            self.cfg.ae_pretrained,
            subfolder="vae",
        ).to(self.device) # type: ignore

        return adapter, unet512, vae

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        Construct the optimizer over both trainable model components.
        """
        return torch.optim.AdamW(
            list(self.adapter.parameters()) + list(self.unet512.parameters()),
            lr=self.cfg.lr,
            betas=self.cfg.optimizer_betas,
            weight_decay=self.cfg.optimizer_weight_decay,
        )

    def _build_noise_scheduler(self) -> DDPMScheduler:
        """
        Construct the diffusion noise scheduler used during training.
        """
        scheduler = DDPMScheduler.from_pretrained(
            self.cfg.scheduler_pretrained,
            subfolder="scheduler",
        )
        scheduler_config = scheduler.config
        set_config_attr(scheduler_config, "prediction_type", "sample")
        return scheduler

    def _prepare_runtime(self) -> None:
        """
        Prepare models, loaders, and optimizer with Accelerate.
        """
        if self.vae is not None:
            (
                self.adapter,
                self.unet512,
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.vae,
            ) = self.accelerator.prepare(
                self.adapter,
                self.unet512,
                self.train_loader,
                self.val_loader,
                self.optimizer,
                self.vae,
            )
        else:
            (
                self.adapter,
                self.unet512,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            ) = self.accelerator.prepare(
                self.adapter,
                self.unet512,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            )

    def _validate_indices(self) -> None:
        """
        Validate that the provided train/validation indices exist and are files.
        """
        if not self.train_index.exists():
            raise FileNotFoundError(f"train_index does not exist: {self.train_index}")

        if not self.train_index.is_file():
            raise ValueError(f"train_index must be a file: {self.train_index}")

        if not self.val_index.exists():
            raise FileNotFoundError(f"val_index does not exist: {self.val_index}")

        if not self.val_index.is_file():
            raise ValueError(f"val_index must be a file: {self.val_index}")

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *,
        train: bool = True,
    ) -> torch.Tensor:
        """
        Run one forward pass for either training or validation.

        Args:
            batch: Tuple of (target_img, z_cond).
            train: Whether to backpropagate the computed loss.

        Returns:
            Scalar loss tensor for the batch.
        """
        target_imgs, z_cond = batch

        z_cond = z_cond.to(self.device, dtype=torch.float16)

        # Compute multi-scale conditional features
        cond_feats = self.adapter(z_cond)

        # Sample random noise
        noise = torch.randn_like(target_imgs)
        num_train_timesteps = get_config_attr(
            self.noise_scheduler.config,
            "num_train_timesteps",
        )
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (target_imgs.size(0),),
            device=self.device,
        ).long()

        # Add noise to target image
        x_noisy = self.noise_scheduler.add_noise(target_imgs, noise, timesteps)  # type: ignore[arg-type]

        # Forward UNet
        x0_pred = self.unet512(x_noisy, timesteps, cond_feats)

        # Compute MSE loss to GT image
        loss = F.mse_loss(x0_pred, target_imgs)

        if train:
            self.accelerator.backward(loss)

        return loss

    @torch.no_grad()
    def validate(
        self,
        *,
        verbose: bool | None = None,
    ) -> float:
        """
        Run the validation loop and return average validation loss.

        Args:
            verbose: Whether to show validation progress.

        Returns:
            Average validation loss.
        """
        if verbose is None:
            verbose = self.verbose

        self.unet512.eval()
        self.adapter.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(
            self.val_loader,
            desc="Validating",
            disable=not verbose,
        ):
            loss = self._step(batch, train=False)
            total_loss += loss.item()
            num_batches += 1

        val_loss = total_loss / max(1, num_batches)

        if verbose:
            self.accelerator.print(f"Validation loss: {val_loss:.4f}")

        return val_loss

    def _train_one_epoch(
        self,
        epoch_idx: int,
        *,
        verbose: bool | None = None,
    ) -> float:
        """
        Run one full training epoch and return the average training loss.

        Args:
            epoch_idx: Zero-based epoch index.
            verbose: Whether to show training progress.

        Returns:
            Average training loss for the epoch.
        """
        if verbose is None:
            verbose = self.verbose

        self.unet512.train()
        self.adapter.train()

        losses: list[float] = []
        prog = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx + 1}/{self.cfg.epochs} [Train]",
            disable=not verbose,
        )

        for batch in prog:
            with self.accelerator.accumulate(self.unet512):
                loss = self._step(batch, train=True)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet512.parameters(), 1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1

            losses.append(loss.item())

            if verbose:
                prog.set_postfix(loss=float(np.mean(losses)))

        train_loss = float(np.mean(losses))
        self.loss_history.append(train_loss)
        return train_loss

    def _build_artifact(
        self,
        *,
        best_epoch: int,
        best_global_step: int,
    ) -> PixelDiffusionArtifact:
        """
        Build the final serializable artifact from the best recovered model state.
        """
        unwrapped_adapter = self.accelerator.unwrap_model(self.adapter)
        unwrapped_unet512 = self.accelerator.unwrap_model(self.unet512)

        return PixelDiffusionArtifact(
            adapter_state_dict=unwrapped_adapter.state_dict(),
            unet_state_dict=unwrapped_unet512.state_dict(),
            adapter_kwargs=self.cfg.adapter_kwargs or {},
            unet_kwargs=self.cfg.unet_kwargs or {},
            scheduler_pretrained=self.cfg.scheduler_pretrained,
            scheduler_num_inference_steps=self.cfg.scheduler_num_inference_steps,
            train_index=str(self.train_index),
            val_index=str(self.val_index),
            bs=self.cfg.train_batch_size,
            lr=self.cfg.lr,
            optimizer_betas=self.cfg.optimizer_betas,
            optimizer_weight_decay=self.cfg.optimizer_weight_decay,
            optimizer_state_dict=self.optimizer.state_dict(),
            best_epoch=best_epoch,
            best_global_step=best_global_step,
        )

    def train(
        self,
        enable_epoch_visualizations: bool = False,
        epoch_visualization_max_batches: int = 2,
        enable_composition_eval: bool = False,
        chart_left_title: str = "Original",
        chart_right_title: str = "Predicted",
        *,
        verbose: bool | None = None,
    ) -> PixelDiffusionArtifact:
        """
        Run the full training loop with validation, early stopping,
        optional qualitative evaluation, checkpoint restore, and
        final artifact creation.

        Args:
            enable_epoch_visualizations: Whether to run qualitative epoch
                visualizations after each validation pass.
            epoch_visualization_max_batches: Maximum number of batches to use
                during epoch visualization.
            enable_composition_eval: Whether to run composition evaluation after
                each validation pass.
            chart_left_title: Left chart title used in composition evaluation.
            chart_right_title: Right chart title used in composition evaluation.
            verbose: Whether to print training progress and show tqdm bars.

        Returns:
            Final pixel diffusion artifact restored from the best checkpoint.
        """
        if verbose is None:
            verbose = self.verbose

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_epoch = 0
        best_global_step = 0

        if verbose:
            self.accelerator.print(
                "Starting Cascade512 training "
                f"(device={self.device}, epochs={self.cfg.epochs}, "
                f"train_batches={len(self.train_loader)}, "
                f"val_batches={len(self.val_loader)})"
            )

        for epoch_idx in range(self.cfg.epochs):
            train_loss = self._train_one_epoch(epoch_idx, verbose=verbose)

            val_loss = self.validate(verbose=verbose)
            self.val_loss_history.append(val_loss)

            if verbose:
                self.accelerator.print(
                    f"[Epoch {epoch_idx + 1}/{self.cfg.epochs}] "
                    f"Train={train_loss:.4f}  Val={val_loss:.4f}"
                )

            # Visualizations
            if enable_epoch_visualizations:
                self.evaluator.visualize_epoch(
                    epoch_idx=epoch_idx,
                    visualization_inference_steps=self.cfg.scheduler_num_inference_steps,
                    max_batches=epoch_visualization_max_batches,
                )

            if enable_composition_eval:
                self.evaluator.eval_composition_batch(
                    epoch_idx=epoch_idx,
                    visualization_inference_steps=self.cfg.scheduler_num_inference_steps,
                    chart_left_title=chart_left_title,
                    chart_right_title=chart_right_title,
                )

            # Improvements
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_epoch = epoch_idx
                best_global_step = self.global_step

                self.accelerator.wait_for_everyone()
                self.accelerator.save_state("ckpt_best")

                if verbose:
                    self.accelerator.print(
                        f"  >> Saved best checkpoint (val={best_val_loss:.4f})"
                    )
            else:
                epochs_without_improvement += 1

                if verbose:
                    self.accelerator.print(
                        f"  >> No improvement "
                        f"({epochs_without_improvement}/{self.cfg.patience})"
                    )

                if epochs_without_improvement >= self.cfg.patience:
                    if verbose:
                        self.accelerator.print("Early stopping triggered.")
                    break

        self.accelerator.wait_for_everyone()
        self.accelerator.load_state("ckpt_best")
        self.accelerator.wait_for_everyone()

        if verbose:
            self.accelerator.print(
                "Finished Cascade512 training "
                f"(best_epoch={best_epoch + 1}, "
                f"best_val_loss={best_val_loss:.4f}, "
                f"best_global_step={best_global_step})"
            )

        return self._build_artifact(
            best_epoch=best_epoch,
            best_global_step=best_global_step,
        )
