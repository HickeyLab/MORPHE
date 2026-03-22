from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

from disco.core.latent_diffusion.artifact import LatentDiffusionArtifact
from disco.core.latent_diffusion.model import BBoxEncoder, CondEncoder, CondEncoder3D, CoordEncoder
from disco.core.latent_diffusion.train.base import LatentTrainStrategy
from disco.core.latent_diffusion.train.train_config import LatentTrainerConfig
from disco.core.pixel_diffusion.models import UNet512
from disco.utils import resolve_device, resolve_dtype


def _get_scaling_factor(vae: AutoencoderKL) -> float:
    return float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

class DiffusionTrainer:
    """
    Trainer for latent diffusion models.

    The trainer owns runtime setup, model/component construction, data loading,
    optimizer creation, and accelerator preparation. Task-specific behavior
    such as dataset construction and architecture selection is delegated to the
    provided ``train_strategy``.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        train_strategy: LatentTrainStrategy,
        cfg: LatentTrainerConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.train_strategy = train_strategy
        self.cfg = cfg

        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)

        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

        self.accelerator = Accelerator(mixed_precision=self.cfg.mixed_precision)

        self.arch_spec = self.train_strategy.build_architecture_spec(cfg=self.cfg)

        self._build_components()
        self._build_dataloaders()
        self._build_optimizer()
        self._prepare_with_accelerator()

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
            dtype=self.dtype,
        )

        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling_factor = _get_scaling_factor(self.vae)

    def _build_dataloaders(self) -> None:
        """
        Build train/validation dataloaders from the strategy-provided datasets.
        """
        train_data, val_data = self.train_strategy.build_dataset(self.root_dir)

        pin_memory = self.device.type == "cuda"

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.train_strategy.train_num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = DataLoader(
            val_data,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.train_strategy.val_num_workers,
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

        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr)

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
            self.optimizer,
        )

        num_modules = len(modules)
        prepared_modules = prepared[:num_modules]
        self.train_loader, self.val_loader, self.optimizer = prepared[num_modules:]

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

    def _train_one_epoch(self, *, epoch: int) -> float:
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
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            with self.accelerator.accumulate(self.unet): # type: ignore
                loss = self.train_strategy.train_step(self, batch)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.cfg.grad_clip is not None:
                    params = self.optimizer.param_groups[0]["params"]
                    self.accelerator.clip_grad_norm_(params, self.cfg.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            lv = float(loss.detach().mean().item())
            losses.append(lv)
            pbar.set_postfix({"loss": float(np.mean(losses))})

        return float(np.mean(losses)) if losses else float("nan")

    def _build_artifact(self) -> LatentDiffusionArtifact:
        """
        Build an in-memory artifact from the trainer's current model states.

        Models are unwrapped before serialization so saved weights come from the
        underlying modules rather than accelerator wrappers.
        """
        unet: UNet512 = self.accelerator.unwrap_model(self.unet)
        cond_encoder: CondEncoder | CondEncoder3D = self.accelerator.unwrap_model(self.cond_encoder)

        coord_state = None
        if self.coord_encoder is not None:
            coord_encoder: CoordEncoder = self.accelerator.unwrap_model(self.coord_encoder)
            coord_state = coord_encoder.state_dict()

        bbox_state = None
        if self.bbox_encoder is not None:
            bbox_encoder: BBoxEncoder = self.accelerator.unwrap_model(self.bbox_encoder)
            bbox_state = bbox_encoder.state_dict()

        return LatentDiffusionArtifact(
            unet_state_dict=unet.state_dict(),
            cond_encoder_state_dict=cond_encoder.state_dict(),
            coord_encoder_state_dict=coord_state,
            bbox_encoder_state_dict=bbox_state,
            architecture=self.arch_spec,
            inference_mode=self.train_strategy.inference_mode,
            img_size=getattr(self.train_strategy, "img_size", None),
        )

    def _save_artifact(self, artifact: LatentDiffusionArtifact, name: str) -> Path:
        """
        Persist an already-built artifact to disk and return its path.
        """
        artifact_path = self.save_dir / name
        artifact.save(artifact_path)
        return artifact_path

    def train(
        self,
        *,
        save_best: bool = True,
        save_last: bool = True,
        checkpoint_every_epochs: int | None = None,
        save_name: str = "latent_diffuser_artifact.pt",
        return_history: bool = False,
    ) -> LatentDiffusionArtifact | tuple[LatentDiffusionArtifact, tuple[list[float], list[float]]]:
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
        best_val = float("inf")
        patience_cnt = 0
        patience = getattr(self.train_strategy, "patience", None)

        best_artifact: LatentDiffusionArtifact | None = None

        save_path = Path(save_name)
        best_name = save_path.name
        best_stem = save_path.stem
        best_suffix = save_path.suffix or ".pt"

        for epoch in range(self.cfg.epochs):
            train_loss = self._train_one_epoch(epoch=epoch)
            self.train_loss_history.append(train_loss)

            val_loss = self.train_strategy.validate_step(trainer=self)
            self.val_loss_history.append(val_loss)

            self.accelerator.print(
                f"Epoch {epoch} -> Train: {train_loss:.6f}  Val: {val_loss:.6f}"
            )

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                self.accelerator.wait_for_everyone()

                best_artifact = self._build_artifact()
                if save_best:
                    self._save_artifact(best_artifact, name=best_name)

            if (
                checkpoint_every_epochs is not None
                and checkpoint_every_epochs > 0
                and (epoch + 1) % checkpoint_every_epochs == 0
            ):
                checkpoint_artifact = self._build_artifact()
                checkpoint_name = f"{best_stem}_epoch_{epoch + 1}{best_suffix}"
                self._save_artifact(checkpoint_artifact, name=checkpoint_name)

            if patience is not None:
                patience_cnt = 0 if improved else patience_cnt + 1
                if patience_cnt >= patience:
                    self.accelerator.print("Early stopping.")
                    break

            if getattr(self.train_strategy, "decay_enabled", False):
                every = getattr(self.train_strategy, "lr_decay_every", None)
                factor = float(getattr(self.train_strategy, "lr_decay_factor", 0.5))
                if every is not None and every > 0 and (epoch + 1) % every == 0:
                    for group in self.optimizer.param_groups:
                        group["lr"] *= factor

        if best_artifact is None:
            best_artifact = self._build_artifact()

        if save_last:
            last_artifact = self._build_artifact()
            last_name = f"{best_stem}_last{best_suffix}"
            self._save_artifact(last_artifact, name=last_name)

        return best_artifact if return_history else best_artifact, (self.train_loss_history, self.val_loss_history)