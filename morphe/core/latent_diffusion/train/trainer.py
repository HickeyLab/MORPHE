from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..artifact import LatentDiffusionArtifact
from ..model import BBoxEncoder, CondEncoder, CondEncoder3D, CoordEncoder
from .base import LatentTrainTask
from .config import LatentTrainerConfig
from ...pixel_diffusion.models import UNet512
from ....utils import resolve_device, resolve_dtype


def _get_scaling_factor(vae: AutoencoderKL) -> float:
    return float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))


@dataclass
class LatentTrainResult:
    artifact: LatentDiffusionArtifact
    train_loss_history: list[float]
    val_loss_history: list[float]


class LatentDiffusionTrainer:
    """
    Trainer for latent diffusion models.

    The trainer owns runtime setup, model/component construction, data loading,
    optimizer creation, and accelerator preparation. Task-specific behavior
    such as dataset construction and architecture selection is delegated to the
    provided ``train_task``.
    """

    def __init__(
        self,
        *,
        input_dir: Path,
        train_task: LatentTrainTask,
        cfg: LatentTrainerConfig,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.train_task = train_task
        self.cfg = cfg
        self.seed = seed

        self.device = resolve_device(device)

        self._set_seed()

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self.accelerator = Accelerator(mixed_precision=self.cfg.mixed_precision)

        self.arch_spec = self.train_task.build_architecture_spec(cfg=self.cfg)

        self._build_components()
        self._build_dataloaders()
        self._prepare_with_accelerator()
        self._build_optimizer()
        
    def encode_with_vae(self, x: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        x = x.to(device=self.accelerator.device, dtype=vae_dtype)
        latents = self.vae.encode(x).latent_dist.sample() # type: ignore
        return latents * self.scaling_factor

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

    def _build_components(self) -> None:
        """
        Build model components from the shared architecture spec.

        The VAE is immediately frozen because it is used only as a pretrained
        encoder/decoder during diffusion training.
        """
        (
            self.unet,
            self.vae,
            self.cond_encoder,
            self.coord_encoder,
            self.bbox_encoder,
            self.noise_scheduler,
        ) = self.arch_spec.build_components(
            device=self.device,
            dtype=torch.float32,
        )

        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling_factor = _get_scaling_factor(self.vae)

    def _build_dataloaders(self) -> None:
        """
        Build train/validation dataloaders from the strategy-provided datasets.
        """
        train_data, val_data = self.train_task.build_dataset(self.input_dir)

        pin_memory = self.device.type == "cuda"

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.train_task.train_num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = DataLoader(
            val_data,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.train_task.val_num_workers,
            pin_memory=pin_memory,
        )

    def _build_optimizer(self) -> None:
        """
        Build the optimizer over all trainable diffusion components.
        """
        params: list[torch.nn.Parameter] = [
            *self.unet.parameters(),
            *self.cond_encoder.parameters(),
        ]

        if self.coord_encoder is not None:
            params.extend(self.coord_encoder.parameters())

        if self.bbox_encoder is not None:
            params.extend(self.bbox_encoder.parameters())

        self.optimizer = self.accelerator.prepare(
            torch.optim.AdamW(params, lr=self.cfg.lr)
        )

    def _prepare_with_accelerator(self) -> None:
        """
        Prepare models, dataloaders, and optimizer with Accelerate.

        Optional modules are handled dynamically so this remains maintainable
        as the runtime evolves.
        """
        modules: list[torch.nn.Module] = [
            self.vae,
            self.unet,
            self.cond_encoder,
        ]

        has_coord = self.coord_encoder is not None
        has_bbox = self.bbox_encoder is not None

        if has_coord:
            coord_encoder = self.coord_encoder
            assert coord_encoder is not None
            modules.append(coord_encoder)

        if has_bbox:
            bbox_encoder = self.bbox_encoder
            assert bbox_encoder is not None
            modules.append(bbox_encoder)

        prepared = self.accelerator.prepare(
            *modules,
            self.train_loader,
            self.val_loader,
        )

        num_modules = len(modules)
        prepared_modules = prepared[:num_modules]
        self.train_loader, self.val_loader = prepared[num_modules:]

        idx = 0
        self.vae = prepared_modules[idx]
        idx += 1

        self.unet = prepared_modules[idx]
        idx += 1

        self.cond_encoder = prepared_modules[idx]
        idx += 1

        if has_coord:
            self.coord_encoder = prepared_modules[idx]
            idx += 1

        if has_bbox:
            self.bbox_encoder = prepared_modules[idx]

    def _train_one_epoch(self, *, epoch: int, verbose: bool) -> float:
        """
        Run one full training epoch and return the mean training loss.
        """
        self.unet.train()
        self.cond_encoder.train()

        if self.coord_encoder is not None:
            self.coord_encoder.train()

        if self.bbox_encoder is not None:
            self.bbox_encoder.train()

        losses: list[float] = []
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            disable=not verbose,
        )

        for batch in pbar:
            with self.accelerator.accumulate(self.unet):  # type: ignore
                loss = self.train_task.train_step(self, batch)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.cfg.grad_clip is not None:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), self.cfg.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            lv = float(loss.detach().mean().item())
            losses.append(lv)
            pbar.set_postfix({"loss": float(np.mean(losses))})

        return float(np.mean(losses)) if losses else float("nan")

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        """
        Return a module's state dict with every tensor moved to CPU.

        The artifact is a portable, serializable object; it should never hold
        GPU-resident tensors because that pins GPU memory for as long as the
        artifact is alive and breaks loading on machines without the same GPU.
        """
        return {key: value.detach().cpu() for key, value in module.state_dict().items()}

    def _build_artifact(self) -> LatentDiffusionArtifact:
        """
        Build an in-memory artifact from the trainer's current model states.

        Models are unwrapped before serialization so saved weights come from the
        underlying modules rather than accelerator wrappers. All parameter
        tensors are copied to CPU so the artifact does not pin GPU memory.
        """
        unet: UNet512 = self.accelerator.unwrap_model(self.unet)
        cond_encoder: CondEncoder | CondEncoder3D = self.accelerator.unwrap_model(self.cond_encoder)

        coord_state = None
        if self.coord_encoder is not None:
            coord_encoder: CoordEncoder = self.accelerator.unwrap_model(self.coord_encoder)
            coord_state = self._cpu_state_dict(coord_encoder)

        bbox_state = None
        if self.bbox_encoder is not None:
            bbox_encoder: BBoxEncoder = self.accelerator.unwrap_model(self.bbox_encoder)
            bbox_state = self._cpu_state_dict(bbox_encoder)

        return LatentDiffusionArtifact(
            unet_state_dict=self._cpu_state_dict(unet),
            cond_encoder_state_dict=self._cpu_state_dict(cond_encoder),
            coord_encoder_state_dict=coord_state,
            bbox_encoder_state_dict=bbox_state,
            architecture=self.arch_spec,
            inference_mode=self.train_task.inference_mode,
            img_size=getattr(self.train_task, "img_size", None),
        )

    def _release(self) -> None:
        """
        Drop GPU-resident training state held on this trainer.

        Called at the end of ``train()`` (including on exception) so the
        trainer does not pin the UNet, VAE, optimizer, accelerator, and
        dataloaders on the GPU after training has finished. The artifact
        returned by ``train()`` already contains CPU copies of the weights,
        so nothing callers care about is lost.

        After this runs the trainer is effectively spent; do not call
        ``train()`` again on the same instance.
        """
        for attr in (
            "unet",
            "vae",
            "cond_encoder",
            "coord_encoder",
            "bbox_encoder",
            "optimizer",
            "accelerator",
            "train_loader",
            "val_loader",
            "noise_scheduler",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def _save_artifact(self, artifact: LatentDiffusionArtifact, save_dir: Path, name: str, verbose: bool) -> Path:
        """
        Persist an already-built artifact to disk and return its path.
        """
        artifact_path = save_dir / name
        artifact.save(artifact_path)

        if verbose:
            print(f"[DiffusionTrainer] Saved artifact to {artifact_path}.")

        return artifact_path

    def train(
        self,
        *,
        save_best: bool = True,
        save_last: bool = True,
        checkpoint_every_epochs: int | None = None,
        save_dir: str | Path = "checkpoints",
        save_name: str = "latent_diffuser_artifact.pt",
        verbose: bool = False,
    ) -> LatentTrainResult:
        """
        Run the full training loop and return the best-performing artifact.

        This method handles:
        - training and validation across epochs
        - early stopping (if defined by the strategy)
        - optional learning rate decay (if enabled by the strategy)
        - checkpointing (best, last, and/or periodic)

        Args:
            save_best:
                If True, saves the best-performing artifact (lowest validation loss)
                to disk using ``save_name``.

            save_last:
                If True, saves the final model state at the end of training to disk
                using a filename of the form ``{stem}_last{suffix}``.

            checkpoint_every_epochs:
                If provided, saves an additional checkpoint every N epochs using
                filenames of the form ``{stem}_epoch_{epoch}{suffix}``.
                The interval is based on completed epochs (1-indexed).

            save_name:
                Base filename used when saving the best checkpoint. This also defines
                the naming scheme for periodic and last checkpoints.

            return_history:
                If True, returns the training and validation loss history along with
                the best artifact.

        Returns:
            Either:
            - The best ``LatentDiffusionArtifact`` (default), or
            - A tuple of:
                (best artifact, (train_loss_history, val_loss_history))
        """
        if verbose:
            print(
                "[DiffusionTrainer] Starting training "
                f"(device={self.device}, seed={self.seed}, "
                f"mixed_precision={self.cfg.mixed_precision})."
            )
            if self.seed is not None:
                print(f"[DiffusionTrainer] Random seed was set to {self.seed}.")
            print("[DiffusionTrainer] Built model components and froze VAE.")
            print(
                "[DiffusionTrainer] Built dataloaders "
                f"(train_batch_size={self.cfg.batch_size}, "
                f"val_batch_size={self.cfg.val_batch_size})."
            )
            print(f"[DiffusionTrainer] Built AdamW optimizer (lr={self.cfg.lr}).")
            print("[DiffusionTrainer] Prepared models, dataloaders, and optimizer with Accelerator.")
            print(f"[DiffusionTrainer] Starting training for {self.cfg.epochs} epochs.")

        try:
            best_val = float("inf")
            patience_cnt = 0
            patience = getattr(self.train_task, "patience", None)

            best_artifact: LatentDiffusionArtifact | None = None

            resolved_save_dir = Path(save_dir)
            resolved_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = Path(save_name)
            best_name = save_path.name
            best_stem = save_path.stem
            best_suffix = save_path.suffix or ".pt"

            for epoch in range(self.cfg.epochs):
                train_loss = self._train_one_epoch(epoch=epoch, verbose=verbose)
                self.train_loss_history.append(train_loss)

                val_loss = self.train_task.validate_step(trainer=self)
                self.val_loss_history.append(val_loss)

                if verbose:
                    self.accelerator.print(
                        f"Epoch {epoch} -> Train: {train_loss:.6f}  Val: {val_loss:.6f}"
                    )

                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    self.accelerator.wait_for_everyone()

                    best_artifact = self._build_artifact()
                    if verbose:
                        print(
                            f"[DiffusionTrainer] New best validation loss at epoch {epoch}: "
                            f"{best_val:.6f}."
                        )
                    if save_best:
                        self._save_artifact(best_artifact, resolved_save_dir, name=best_name, verbose=verbose)

                if (
                    checkpoint_every_epochs is not None
                    and checkpoint_every_epochs > 0
                    and (epoch + 1) % checkpoint_every_epochs == 0
                ):
                    checkpoint_artifact = self._build_artifact()
                    checkpoint_name = f"{best_stem}_epoch_{epoch + 1}{best_suffix}"
                    self._save_artifact(checkpoint_artifact, resolved_save_dir, name=checkpoint_name, verbose=verbose)

                if patience is not None:
                    patience_cnt = 0 if improved else patience_cnt + 1
                    if patience_cnt >= patience:
                        self.accelerator.print("Early stopping.")
                        if verbose:
                            print(
                                f"[DiffusionTrainer] Early stopping triggered at epoch {epoch} "
                                f"with patience={patience}."
                            )
                        break

                if getattr(self.train_task, "decay_enabled", False):
                    every = getattr(self.train_task, "lr_decay_every", None)
                    factor = float(getattr(self.train_task, "lr_decay_factor", 0.5))
                    if every is not None and every > 0 and (epoch + 1) % every == 0:
                        for group in self.optimizer.param_groups:
                            group["lr"] *= factor
                        if verbose:
                            print(
                                "[DiffusionTrainer] Decayed learning rate by factor "
                                f"{factor} at epoch {epoch}."
                            )

            if best_artifact is None:
                best_artifact = self._build_artifact()

            if save_last:
                last_artifact = self._build_artifact()
                last_name = f"{best_stem}_last{best_suffix}"
                self._save_artifact(last_artifact, resolved_save_dir, name=last_name, verbose=verbose)

            if verbose:
                print("[DiffusionTrainer] Training complete.")

            return LatentTrainResult(
                artifact=best_artifact,
                train_loss_history=self.train_loss_history,
                val_loss_history=self.val_loss_history
            )
        finally:
            self._release()