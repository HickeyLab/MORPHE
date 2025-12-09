import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL  # if used for composition inference

from data.dataset_cascade import PrecomputedCascadeDataset
from models.latent_adapter import LatentAdapter
from models.unet512 import UNet512
from utils.composition import infer_cell_map, compute_type_distribution


# =====================================================================
# Cascade512Trainer
# =====================================================================
class Cascade512Trainer:
    """
    Stage-2 (512×512) cascade diffusion trainer:

    - Loads precomputed latents (z_cond)
    - Trains UNet512 to denoise pixel-space image
    - Supports validation, sampling, composition evaluation
    - Supports fp16 + multi-GPU via Accelerate
    """

    def __init__(self, train_index, val_index, bs=4, lr=1e-5):

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
        self.adapter = LatentAdapter(cz=4, cond_ch=64)
        self.unet512 = UNet512(base_ch=128, cond_ch=64, time_dim=256)

        # -------------------------------
        # Optimizer
        # -------------------------------
        self.optimizer = torch.optim.AdamW(
            list(self.adapter.parameters()) + list(self.unet512.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
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
        self.vis_dir = "visualization_savepath"
        os.makedirs(self.vis_dir, exist_ok=True)

    # ==================================================================
    # One training/validation step
    # ==================================================================
    def _step(self, batch, train=True):
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
    def visualize_epoch(self, epoch_idx, max_batches=1, steps_stage2=50):
        self.unet512.eval()
        self.adapter.eval()

        os.makedirs(self.vis_dir, exist_ok=True)

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

    # ==================================================================
    # Composition Evaluation
    # ==================================================================
    @torch.no_grad()
    def eval_composition_batch(
        self, ae_model, index, steps_stage2=50, save_dir="comp_eval"
    ):
        self.unet512.eval()
        self.adapter.eval()
        os.makedirs(save_dir, exist_ok=True)

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
            type_pred = infer_cell_map(pred_imgs[i], ae_model)
            type_orig = infer_cell_map(target_vis[i], ae_model)

            dist_pred = compute_type_distribution(type_pred.squeeze().cpu().numpy())
            dist_orig = compute_type_distribution(type_orig.squeeze().cpu().numpy())

            fr_pred.append(dist_pred)
            fr_orig.append(dist_orig)

        # Save plots
        xs = np.arange(len(fr_pred[0]))

        for i in range(B):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].bar(xs, fr_orig[i])
            axes[0].set_title("Original Composition")

            axes[1].bar(xs, fr_pred[i])
            axes[1].set_title("Predicted Composition")

            plt.tight_layout()
            out_path = os.path.join(save_dir, f"comp_eval_{index}_img{i}.png")
            plt.savefig(out_path)
            plt.close()

            self.accelerator.print(f"[CompEval] Saved {out_path}")

    # ==================================================================
    # Main Training Loop
    # ==================================================================
    def train(self, epochs=30, patience=5, vis_steps_stage2=50):

        # Optionally load autoencoder model
        ae = AutoencoderKL.from_pretrained("your_ae_path").to(self.device).eval()

        best = float("inf")
        bad = 0

        for ep in range(1, epochs + 1):

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

            # Visualization
            self.visualize_epoch(ep, max_batches=2, steps_stage2=vis_steps_stage2)

            # Composition eval
            self.eval_composition_batch(ae, ep, steps_stage2=vis_steps_stage2)

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
