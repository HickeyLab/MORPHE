import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

from utils.mask_utils import create_latent_mask
from utils.plot_utils import plot_loss

from models.cond_encoder import CondEncoder
from models.coord_encoder import CoordEncoder

from data.stage1_dataset import Stage1Dataset
from torch.utils.data import DataLoader


class LatentTrainer:
    def __init__(self, pretrained="runwayml/stable-diffusion-v1-5"):
        self.accelerator = Accelerator(mixed_precision="fp16")

        train = Stage1Dataset("train_data")
        val = Stage1Dataset("val_data")

        self.train_loader = DataLoader(train, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=4)

        # Load pretrained modules
        self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")

        self.coord_enc = CoordEncoder()
        self.cond_enc = CondEncoder()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters()) +
            list(self.coord_enc.parameters()) +
            list(self.cond_enc.parameters()),
            lr=2e-5
        )

        components = [
            self.vae, self.unet, self.coord_enc, self.cond_enc,
            self.train_loader, self.val_loader, self.optimizer
        ]
        (
            self.vae, self.unet, self.coord_enc, self.cond_enc,
            self.train_loader, self.val_loader, self.optimizer
        ) = self.accelerator.prepare(*components)

        self.vae.requires_grad_(False)
        self.scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

        self.train_loss = []
        self.val_loss = []

    # ============ train step ============
    def train_step(self, batch):
        masked_img, target_img, bbox = batch

        with torch.no_grad():
            target_lat = self.vae.encode(target_img).latent_dist.sample()
            target_lat = target_lat * self.vae.config.scaling_factor

        mask = create_latent_mask(bbox, target_lat.shape, target_lat.device)

        noise = torch.randn_like(target_lat)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (target_lat.size(0),), device=target_lat.device)

        noisy = self.scheduler.add_noise(target_lat * mask, noise * mask, t)
        noisy = target_lat * (1 - mask) + noisy

        with torch.no_grad():
            masked_lat = self.vae.encode(masked_img).latent_dist.sample()
            masked_lat = masked_lat * self.vae.config.scaling_factor

        cond_bbox = self.coord_enc(bbox)
        cond_tokens = torch.cat([
            self.cond_enc(masked_lat),
            cond_bbox.unsqueeze(1).expand(-1, 64, -1)
        ], dim=-1)

        pred = self.unet(noisy, t, encoder_hidden_states=cond_tokens).sample

        loss = F.mse_loss(pred * mask, noise * mask)
        return loss

    # ============ validation ============
    def validate(self):
        self.unet.eval()
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                total += self.train_step(batch).item()
        return total / len(self.val_loader)

    # ============ main train loop ============
    def train(self, epochs=10):
        best_val = 1e9
        patience = 0

        for ep in range(epochs):
            self.unet.train()
            ep_losses = []

            for batch in tqdm(self.train_loader, desc=f"Epoch {ep}"):
                with self.accelerator.accumulate():
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                ep_losses.append(loss.item())

            train_avg = sum(ep_losses) / len(ep_losses)
            val_avg = self.validate()

            self.train_loss.append(train_avg)
            self.val_loss.append(val_avg)

            print(f"Epoch {ep}: train={train_avg:.4f} val={val_avg:.4f}")

            if val_avg < best_val:
                best_val = val_avg
                patience = 0
                self.accelerator.save_state("drive/MyDrive/checkpoint-merfish")
            else:
                patience += 1

            if patience >= 5:
                print("Early stopping.")
                break

            plot_loss(self.train_loss, self.val_loss)
