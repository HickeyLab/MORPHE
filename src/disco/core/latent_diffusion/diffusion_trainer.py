import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from accelerate import Accelerator
from tqdm import tqdm
from disco.core.latent_diffusion.artifact import LatentDiffuserArtifact

from disco.core.latent_diffusion.model import CondEncoder, CoordEncoder, CondEncoder3D
from disco.core.latent_diffusion.strategy.base import DiffusionStrategy
from disco.viz.loss_curve import plot_loss_curve

def _get_scaling_factor(vae: AutoencoderKL) -> float:
    return float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215))

class DiffusionTrainer:
    def __init__(
        self,
        *,
        strategy: DiffusionStrategy,
        root_dir: Path,
        save_dir: str = "checkpoints",
        save_safetensors: bool = True,
        save_best_only: bool = True,
        pretrained: str = "runwayml/stable-diffusion-v1-5",
        lr: float = 2e-5,
        mixed_precision: str = "fp16",
        grad_clip: float = 1.0,
        patience: int = 5,
        cond_encoder_kwargs: dict[str, int],
        coord_encoder_kwargs: dict[str, int] | None,
        batch_size: int = 8,
        val_batch_size: int = 8
    ):
        self.strategy = strategy
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)
        self.save_safetensors = bool(save_safetensors)
        
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        
        train_data, val_data = strategy.build_dataset(root_dir)
        self.train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=self.strategy.train_num_workers, 
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_data, 
            batch_size=val_batch_size,
            shuffle=False, 
            num_workers=self.strategy.val_num_workers, 
            pin_memory=True
        )

        # Load pretrained modules
        # TODO: ASK ABOUT WHETHER I SHOULD LOAD PRETRAINED STUFF FROM UNET AND CONDENCODER IN 3D?
        self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
        
        latent_c = self.vae.config.latent_channels
        self.cond_proj = CondEncoder(
            in_channels=latent_c if not getattr(cond_encoder_kwargs, "cond_in_channels", None) else getattr(cond_encoder_kwargs, "cond_in_channels", None),
            **cond_encoder_kwargs
        )
        if strategy.requires_coord_encoder:
            self.coord_encoder = CoordEncoder(**coord_encoder_kwargs)
        
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling_factor = _get_scaling_factor(self.vae)
    
        # Optimizer
        params = (
            list(self.unet.parameters())
            + list(self.cond_proj.parameters())
            + (list(self.coord_encoder.parameters()) if strategy.requires_coord_encoder else [])
        )

        self.optimizer = torch.optim.AdamW(params, lr=lr)
        
         # prepare components with accelerator
        if strategy.requires_coord_encoder:
            (
                self.vae,
                self.unet,
                self.cond_proj,
                self.coord_encoder,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            ) = self.accelerator.prepare(
                self.vae,
                self.unet,
                self.cond_proj,
                self.coord_encoder,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            )
        else:
            (
                self.vae,
                self.unet,
                self.cond_proj,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            ) = self.accelerator.prepare(
                self.vae,
                self.unet,
                self.cond_proj,
                self.train_loader,
                self.val_loader,
                self.optimizer,
            )


        os.makedirs(self.save_dir, exist_ok=True)
        

    def _train_one_epoch(self, *, epoch: int) -> float:
        self.unet.train()
        losses: list[float] = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            with self.accelerator.accumulate(self.unet):
                loss = self.strategy.train_step(self, batch)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients and self.grad_clip is not None:
                    # clip whatever optimizer is actually stepping
                    params = self.optimizer.param_groups[0]["params"]
                    self.accelerator.clip_grad_norm_(params, self.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            lv = float(loss.detach().mean().item())
            losses.append(lv)
            pbar.set_postfix({"loss": float(np.mean(losses))})

        return float(np.mean(losses)) if losses else float("nan")
    
    
    def _save_best_checkpoint(self, save_dir: Path):
        save_dir.mkdir(exist_ok=True)

        coord_state = None
        coord_kwargs = None
        if self.strategy.requires_coord_encoder:
            coord_state = self.coord_encoder.state_dict()
            coord_kwargs = dict(getattr(self.strategy, "coord_encoder_kwargs", {}) or {})

        artifact = LatentDiffuserArtifact(
            unet_state_dict=self.unet.state_dict(),
            cond_encoder_state_dict=self.cond_proj.state_dict(),
            coord_encoder_state_dict=coord_state,
            unet_config=self.unet.config.to_dict(),
            cond_encoder_kwargs=dict(getattr(self.strategy, "cond_encoder_kwargs", {}) or {}),
            coord_encoder_kwargs=coord_kwargs,
        )

        artifact_path = save_dir / "latent_diffuser_artifact.pt"
        artifact.save(artifact_path)


    def train(
        self, 
        *, 
        epochs: int = 20,
        show_loss_curve: bool = False,
        figsize: tuple[int, int] = (7, 5),
        save_checkpoints: bool = True,
    ) -> None:
        best_val = float("inf")
        patience_cnt = 0
        patience = getattr(self.strategy, "patience", None)

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(epoch=epoch)
            self.train_loss_history.append(train_loss)

            val_loss = self.strategy.validate_step(trainer=self)
            self.val_loss_history.append(val_loss)

            print(f"Epoch {epoch} -> Train: {train_loss:.6f}  Val: {val_loss:.6f}")

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                self.accelerator.wait_for_everyone()
                
                if save_checkpoints:
                    self._save_best_checkpoint(self.save_dir)

            # early stopping only if enabled on the strategy
            if patience is not None:
                if improved:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        print("Early stopping.")
                        break

            # lr decay only if enabled on the strategy
            decay_enabled = getattr(self.strategy, "decay_enabled", False)
            if decay_enabled: 
                every = getattr(self.strategy, "lr_decay_every", None)
                factor = float(getattr(self.strategy, "lr_decay_factor", 0.5))
                if every is not None and epoch % every == 0:
                    for g in self.optimizer.param_groups:
                        g["lr"] *= factor

            if show_loss_curve:
                plot_loss_curve(
                    train_losses=self.train_loss_history,
                    val_losses=self.val_loss_history,
                    figsize=figsize,
                    title="Loss Curve",
                    save_path=self.save_dir
                )
