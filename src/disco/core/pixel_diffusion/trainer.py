from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

from Pixel_Diffusion_Decoder.data.dataset_cascade import PrecomputedCascadeDataset
from disco.core.pixel_diffusion.artifact import PixelDiffusionArtifact
from disco.core.pixel_diffusion.config import Cascade512TrainerConfig
from disco.core.pixel_diffusion.evaluator import Cascade512Evaluator
from disco.core.pixel_diffusion.models import LatentAdapter, UNet512
from disco.utils import get_config_attr, set_config_attr


class Cascade512Trainer:
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
        cfg: Cascade512TrainerConfig,
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
        """
        self.cfg = cfg
        self.global_step: int = 0

        self.train_index = Path(train_index)
        self.val_index = Path(val_index)
        self._validate_indices()

        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        # AMP + distributed manager
        self.accelerator = self._build_accelerator()
        self.device = self.accelerator.device

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.adapter, self.unet512, self.vae = self._build_models()
        self.optimizer = self._build_optimizer()
        self.noise_scheduler = self._build_noise_scheduler()

        # -------------------------------
        # Prepare all for Accelerator
        # -------------------------------
        self._prepare_runtime()

        # -------------------------------
        # Evaluator
        # -------------------------------
        self.evaluator = Cascade512Evaluator(
            adapter=self.adapter,
            unet512=self.unet512,
            noise_scheduler=self.noise_scheduler,
            val_loader=self.val_loader,
            cfg=self.cfg,
            device=self.device,
            accelerator=self.accelerator,
            vae=self.vae,
        )

    def _build_accelerator(self) -> Accelerator:
        """
        Build the Accelerate runtime manager.
        """
        return Accelerator(mixed_precision="fp16")

    def _build_dataloaders(
        self,
    ) -> tuple[
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Construct train and validation dataloaders from the precomputed indices.
        """
        # -------------------------------
        # Dataset
        # -------------------------------
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
        # -------------------------------
        # Model
        # -------------------------------
        adapter_kwargs = self.cfg.adapter_kwargs or {}
        unet_kwargs = self.cfg.unet_kwargs or {}

        adapter = LatentAdapter(**adapter_kwargs)  # type: ignore[arg-type]
        unet512 = UNet512(**unet_kwargs)

        vae: AutoencoderKL | None = None
        if self.cfg.enable_epoch_visualizations:
            vae = AutoencoderKL.from_pretrained(
                self.cfg.ae_pretrained,
                subfolder="vae",
            ).to(self.device)

        return adapter, unet512, vae

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        Construct the optimizer over both trainable model components.
        """
        # -------------------------------
        # Optimizer
        # -------------------------------
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
        # -------------------------------
        # Scheduler
        # -------------------------------
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

    # ==================================================================
    # One training/validation step
    # ==================================================================
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

    # ==================================================================
    # Validation loop
    # ==================================================================
    @torch.no_grad()
    def validate(self) -> float:
        """
        Run the validation loop and return average validation loss.
        """
        self.unet512.eval()
        self.adapter.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            loss = self._step(batch, train=False)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _train_one_epoch(self, epoch_idx: int) -> float:
        """
        Run one full training epoch and return the average training loss.
        """
        self.unet512.train()
        self.adapter.train()

        losses: list[float] = []
        prog = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} [Train]")

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
            prog.set_postfix(loss=np.mean(losses))

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
            unet512_state_dict=unwrapped_unet512.state_dict(),
            adapter_kwargs=self.cfg.adapter_kwargs or {},
            unet_kwargs=self.cfg.unet_kwargs or {},
            train_index=str(self.train_index),
            val_index=str(self.val_index),
            bs=self.cfg.train_batch_size,
            lr=self.cfg.lr,
            ae_pretrained=self.cfg.ae_pretrained,
            enable_epoch_visualizations=self.cfg.enable_epoch_visualizations,
            optimizer_betas=self.cfg.optimizer_betas,
            optimizer_weight_decay=self.cfg.optimizer_weight_decay,
            optimizer_state_dict=self.optimizer.state_dict(),
            epoch=best_epoch,
            global_step=best_global_step,
        )

    # ==================================================================
    # Main Training Loop
    # ==================================================================
    def train(self) -> PixelDiffusionArtifact:
        """
        Run the full training loop with validation, early stopping,
        optional qualitative evaluation, checkpoint restore, and
        final artifact creation.
        """
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_epoch = 0
        best_global_step = 0

        for epoch_idx in range(self.cfg.epochs):
            train_loss = self._train_one_epoch(epoch_idx)

            val_loss = self.validate()
            self.val_loss_history.append(val_loss)

            self.accelerator.print(
                f"[Epoch {epoch_idx}] Train={train_loss:.4f}  Val={val_loss:.4f}"
            )

            self.evaluator.maybe_run_epoch_evaluation(epoch_idx=epoch_idx)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_epoch = epoch_idx
                best_global_step = self.global_step

                self.accelerator.wait_for_everyone()
                self.accelerator.save_state("ckpt_best")
                self.accelerator.print(
                    f"  >> Saved best checkpoint (val={best_val_loss:.4f})"
                )
            else:
                epochs_without_improvement += 1
                self.accelerator.print(
                    f"  >> No improvement ({epochs_without_improvement}/{self.cfg.patience})"
                )
                if epochs_without_improvement >= self.cfg.patience:
                    self.accelerator.print("Early stopping triggered.")
                    break

        self.accelerator.wait_for_everyone()
        self.accelerator.load_state("ckpt_best")
        self.accelerator.wait_for_everyone()

        return self._build_artifact(
            best_epoch=best_epoch,
            best_global_step=best_global_step,
        )