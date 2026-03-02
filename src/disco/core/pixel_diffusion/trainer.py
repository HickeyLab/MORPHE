import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.utils as vutils

from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL
from disco.core.autoencoder.artifact import AutoencoderArtifact

from disco.core.pixel_diffusion.data import PrecomputedCascadeDataset
from disco.core.pixel_diffusion.models import LatentAdapter, UNet512
from disco.viz.cell_diagram_chart import save_side_by_side_barplot

        
class Cascade512Trainer:
    """
    Stage-2 (512×512) cascade diffusion trainer:

    - Loads precomputed latents (z_cond)
    - Trains UNet512 to denoise pixel-space image
    - Supports validation, sampling, composition evaluation
    - Supports fp16 + multi-GPU via Accelerate
    """

    def __init__(
        self,
        *
        train_index: str,
        val_index: str,
        bs: int = 4,
        lr: float = 1e-5,
        vis_dir: str | Path = "visualizations",
        ae_pretrained: str = "runwayml/stable-diffusion-v1-5",
        enable_epoch_visualiations: bool = False,
        adapter_kwargs: dict[str, int] | None = None,
        unet_kwargs: dict[str, int] | None = None,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
        optimizer_weight_decay: float = 1e-5
    ):
        vis_dir = self._validate_init_args(
            train_index, 
            val_index, 
            bs, 
            lr, 
            vis_dir,
            ae_pretrained,
            enable_epoch_visualiations,
            adapter_kwargs,
            unet_kwargs,
            optimizer_betas,
            optimizer_weight_decay
        )

        # AMP + distributed manager
        self.accelerator = Accelerator(mixed_precision="fp16")
        self.device = self.accelerator.device

        self.loss_history = []
        self.val_loss_history = []

        # -------------------------------
        # Dataset
        # -------------------------------
        self.train_loader = DataLoader(
            PrecomputedCascadeDataset(train_index),
            batch_size=bs,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            PrecomputedCascadeDataset(val_index),
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # -------------------------------
        # Model
        # -------------------------------
        self.adapter = LatentAdapter(**adapter_kwargs)
        self.unet512 = UNet512(**unet_kwargs)
        self.vae = AutoencoderArtifact.load(ae_pretrained) if enable_epoch_visualiations else None

        # -------------------------------
        # Optimizer
        # -------------------------------
        self.optimizer = torch.optim.AdamW(
            list(self.adapter.parameters()) + list(self.unet512.parameters()),
            lr=lr,
            betas=optimizer_betas,
            weight_decay=optimizer_weight_decay,
        )

        # -------------------------------
        # Scheduler
        # -------------------------------
        self.scheduler2 = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )
        self.scheduler2.config.prediction_type = "sample"

        # -------------------------------
        # Prepare all for Accelerator
        # -------------------------------
        if enable_epoch_visualiations:
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
                self.vae
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
        # -------------------------------
        # Visualization paths
        # -------------------------------
        self.vis_dir = vis_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        
        self.enable_epoch_visualiations = enable_epoch_visualiations
        
    def _validate_bool(self, name: str, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool, got {type(value).__name__}")

    def _validate_positive_int(self, name: str, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")

    def _validate_positive_number(self, name: str, value: float):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")

    def _validate_pathlike_str(self, name: str, value: str | Path):
        if not (isinstance(value, str) or isinstance(value, Path)) or not value:
            raise ValueError(f"{name} must be a non-empty str or Path")

    def _validate_dict(self, name: str, value: dict):
        if value is None:
            return
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be dict or None, got {type(value).__name__}")

    def _validate_optional_int(self, name: str, value: int | None):
        if value is None:
            return
        if not isinstance(value, int):
            raise TypeError(f"{name} must be int or None, got {type(value).__name__}")

    def _validate_optional_float(self, name: str, value: float | None):
        if value is None:
            return
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be float or None, got {type(value).__name__}")

    def _validate_init_args(
        self,
        train_index: str,
        val_index: str,
        bs: int,
        lr: float,
        vis_dir: str | Path,
        ae_pretrained: str | None,
        enable_epoch_visualiations: bool, 
        adapter_kwargs: dict | None = None,
        unet_kwargs: dict | None = None,
    ) -> list[Path]:
        self._validate_pathlike_str("train_index", train_index)
        self._validate_pathlike_str("val_index", val_index)
        self._validate_positive_int("bs", bs)
        self._validate_positive_number("lr", lr)
        self._validate_pathlike_str("vis_dir", vis_dir)
        vis_dir = Path(vis_dir)
        
        self._validate_bool("enable_epoch_visualizations", enable_epoch_visualiations)
        if enable_epoch_visualiations:
            self._validate_pathlike_str("ae_path", ae_pretrained)

        self._validate_dict("adapter_kwargs", adapter_kwargs)
        self._validate_dict("unet_kwargs", unet_kwargs)
            
        return [vis_dir]

    def _validate_train_args(self, epochs: int, patience: int, vis_steps_stage2: int):
        self._validate_positive_int("epochs", epochs)
        self._validate_positive_int("patience", patience)
        self._validate_positive_int("vis_steps_stage2", vis_steps_stage2)

    def _validate_visualize_args(self, epoch_idx: int, max_batches: int, steps_stage2: int):
        self._validate_positive_int("epoch_idx", epoch_idx)
        self._validate_positive_int("max_batches", max_batches)
        self._validate_positive_int("steps_stage2", steps_stage2)

    def _validate_eval_comp_args(self, index: int, steps_stage2: int, save_dir: str):
        self._validate_positive_int("index", index)
        self._validate_positive_int("steps_stage2", steps_stage2)
        self._validate_pathlike_str("save_dir", save_dir)

    # ==================================================================
    # One training/validation step
    # ==================================================================
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], train: bool = True):
        """
        batch = (target_img, z_cond)
        """
        target_imgs, z_cond = batch

        z_cond = z_cond.to(self.device, dtype=torch.float16)

        # Compute multi-scale conditional features
        cond_feats = self.adapter(z_cond)

        # Sample random noise
        noise = torch.randn_like(target_imgs)
        timesteps = torch.randint(
            0,
            self.scheduler2.config.num_train_timesteps,
            (target_imgs.size(0),),
            device=self.device
        ).long()

        # Add noise to target image
        x_noisy = self.scheduler2.add_noise(target_imgs, noise, timesteps)

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
    def validate(self):
        self.unet512.eval()
        self.adapter.eval()

        total, n = 0.0, 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            loss = self._step(batch, train=False)
            total += loss.item()
            n += 1

        return total / max(1, n)

    # ==================================================================
    # Visualize sampled predictions each epoch
    # ==================================================================
    @torch.no_grad()
    def visualize_epoch(
        self,
        epoch_idx: int,
        max_batches: int = 1,
        steps_stage2: int = 50,
    ):
        self._validate_visualize_args(epoch_idx, max_batches, steps_stage2)

        self.unet512.eval()
        self.adapter.eval()

        self.scheduler2.set_timesteps(steps_stage2, device=self.device)

        saved = 0
        grids = []
        for batch in self.val_loader:

            target_imgs, z_cond = batch
            B = target_imgs.size(0)

            z_cond = z_cond.to(self.device, dtype=torch.float16)
            cond_feats = self.adapter(z_cond)

            # Start from pure noise
            x = torch.randn(
                B, 3, target_imgs.shape[2], target_imgs.shape[3],
                device=self.device
            ) * self.scheduler2.init_noise_sigma

            for t in self.scheduler2.timesteps:
                t_batch = torch.full((B,), int(t), device=self.device, dtype=torch.long)
                x0_pred = self.unet512(x, t_batch, cond_feats)
                x = self.scheduler2.step(x0_pred, t, x).prev_sample

            pred = (x.clamp(-1, 1) + 1) / 2
            target_vis = (target_imgs.clamp(-1, 1) + 1) / 2

            triplet = torch.cat([target_vis, pred], dim=0)
            grid = vutils.make_grid(triplet, nrow=B, padding=2)

            grids.append(grid)
            saved += 1
            if saved >= max_batches:
                break

        if grids:
            final_grid = torch.cat(grids, dim=1) if len(grids) > 1 else grids[0]
            out_path = os.path.join(self.vis_dir, f"epoch_{epoch_idx:03d}.png")
            vutils.save_image(final_grid, out_path)
            self.accelerator.print(f"[Visualize] Saved {out_path}")
            
    def _infer_cell_map(
        img: torch.Tensor,
        ae_model: AutoencoderKL,
    ) -> torch.Tensor:
        """
        Runs autoencoder → returns type map
        """
        with torch.no_grad():
            return ae_model(img.unsqueeze(0))


    def _compute_type_distribution(
        type_map: np.ndarray | torch.Tensor,
        num_types: int = 25,
    ) -> np.ndarray:
        """
        Compute histogram of cell types
        """
        if isinstance(type_map, torch.Tensor):
            type_map = type_map.detach().cpu().numpy()

        flat = type_map.flatten()
        hist, _ = np.histogram(flat, bins=num_types, range=(0, num_types))
        return hist / (hist.sum() + 1e-8)

    # ==================================================================
    # Composition Evaluation
    # ==================================================================
    @torch.no_grad()
    def eval_composition_batch(
        self,
        ae_model: AutoencoderKL,
        *
        index: int,
        steps_stage2: int = 50,
        save_dir: str = "comp_eval",
        chart_left_title: str = "Original Composition",
        chart_right_title: str = "Predicted Composition"
    ):
        self._validate_eval_comp_args(index, steps_stage2, save_dir)

        self.unet512.eval()
        self.adapter.eval()

        self.scheduler2.set_timesteps(steps_stage2, device=self.device)

        batch = next(iter(self.val_loader))
        target_imgs, z_cond = batch
        B = target_imgs.size(0)

        z_cond = z_cond.to(self.device, dtype=torch.float16)
        cond_feats = self.adapter(z_cond)

        # Start from noise
        x = torch.randn(
            B, 3, target_imgs.shape[2], target_imgs.shape[3], device=self.device
        ) * self.scheduler2.init_noise_sigma

        for t in self.scheduler2.timesteps:
            t_batch = torch.full((B,), int(t), device=self.device, dtype=torch.long)
            x0_pred = self.unet512(x, t_batch, cond_feats)
            x = self.scheduler2.step(x0_pred, t, x).prev_sample

        pred_imgs = (x.clamp(-1, 1) + 1) / 2
        target_vis = (target_imgs.clamp(-1, 1) + 1) / 2

        # Compute compositions
        fr_pred, fr_orig = [], []

        for i in range(B):
            type_pred = self._infer_cell_map(pred_imgs[i], ae_model)
            type_orig = self._infer_cell_map(target_vis[i], ae_model)

            dist_pred = self._compute_type_distribution(type_pred.squeeze().cpu().numpy())
            dist_orig = self._compute_type_distribution(type_orig.squeeze().cpu().numpy())

            fr_pred.append(dist_pred)
            fr_orig.append(dist_orig)
        
        # Save plots
        xs = np.arange(len(fr_pred[0]))

        for i in range(B):
            out_path = os.path.join(save_dir, f"comp_eval_{index}_img{i}.png")
            save_side_by_side_barplot(
                xs=xs,
                left_vals=np.asarray(fr_orig[i]),
                right_vals=np.asarray(fr_pred[i]),
                left_title=chart_left_title,
                right_title=chart_right_title,
                out_path=out_path,
            )
            self.accelerator.print(f"[CompEval] Saved {out_path}")

    # ==================================================================
    # Main Training Loop
    # ==================================================================
    def train(
        self,
        epochs: int = 30,
        patience: int = 5,
        vis_steps_stage2: int = 50,
        enable_visualize: bool = True,
        enable_composition_eval: bool = True,
    ):
        self._validate_train_args(epochs, patience, vis_steps_stage2)
        self._validate_bool("enable_visualize", enable_visualize)
        self._validate_bool("enable_composition_eval", enable_composition_eval)

        best = float("inf")
        bad = 0
        for ep in range(0, epochs):
            self.unet512.train()
            self.adapter.train()

            losses = []
            prog = tqdm(self.train_loader, desc=f"Epoch {ep} [Train]")

            for batch in prog:
                with self.accelerator.accumulate(self.unet512):
                    loss = self._step(batch, train=True)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet512.parameters(), 1.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                losses.append(loss.item())
                prog.set_postfix(loss=np.mean(losses))

            train_loss = float(np.mean(losses))
            self.loss_history.append(train_loss)

            # Validation
            val_loss = self.validate()
            self.val_loss_history.append(val_loss)

            self.accelerator.print(
                f"[Epoch {ep}] Train={train_loss:.4f}  Val={val_loss:.4f}"
            )
            
            if self.enable_epoch_visualiations:
                # Visualization
                self.visualize_epoch(
                    ep,
                    max_batches=2,
                    steps_stage2=vis_steps_stage2,
                    enable_plots=enable_visualize,
                )

                # Composition eval
                self.eval_composition_batch(
                    self.vae,
                    ep,
                    steps_stage2=vis_steps_stage2,
                    enable_plots=enable_composition_eval,
                )

            # -------------------------
            # Early stopping + checkpoint
            # -------------------------
            if val_loss < best - 1e-4:
                best = val_loss
                bad = 0

                self.accelerator.wait_for_everyone()
                self.accelerator.save_state("ckpt_best")
                self.accelerator.print(f"  >> Saved best checkpoint (val={best:.4f})")
            else:
                bad += 1
                self.accelerator.print(f"  >> No improvement ({bad}/{patience})")
                if bad >= patience:
                    self.accelerator.print("Early stopping triggered.")
                    break
