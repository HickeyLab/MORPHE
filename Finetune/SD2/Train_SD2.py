"""
sd2_inpaint_finetune_cuda_no_text.py

Stable Diffusion 2.1 inpainting trainer WITHOUT text input.
Uses ONLY image-context embedding from OpenCLIP Vision model.
UNet is adapted to accept (noisy_latents, masked_latents, mask) as input.
"""

import os
import math
import random
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

# -------------------------
# CONFIG
# -------------------------
SD2_BASE = "Manojb/stable-diffusion-2-1-base"
VAE_REPO = SD2_BASE
CLIP_VISION = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

TRAIN_ROOT = "drive/MyDrive/Trainset"
VAL_ROOT   = "drive/MyDrive/202509_CURRENT_Diff_VAL_set"

IMG_SIZE = 512
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
EPOCHS = 20
LR = 2e-5
PATIENCE = 4
MIXED_PRECISION = "fp16"
LATENT_MASK_THRESHOLD = 0.5
SAVE_DIR = "drive/MyDrive/sd2/"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------
# Dataset
# -------------------------
class OutpaintDataset(Dataset):
    def __init__(self, root_dir, img_size=512, masks_per_image=40):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                      if f.lower().endswith(("jpg","png","jpeg"))]
        self.masks_per_image = masks_per_image

        self.tf = transforms.Compose([
            transforms.RandomChoice([
                  transforms.Lambda(lambda x: x),
                  transforms.Lambda(lambda x: TF.rotate(x, 90)),
                  transforms.Lambda(lambda x: TF.rotate(x, 180)),
                  transforms.Lambda(lambda x: TF.rotate(x, 270))
            ]),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files) * self.masks_per_image

    def _gen_bbox(self):
        if random.random() < 0.5:
            ar = random.uniform(1,33)
        else:
            ar = random.uniform(0.03,1)
        area = random.uniform(0.1,0.33)**2
        w = min(math.sqrt(area*ar), 0.99)
        h = min(math.sqrt(area/ar), 0.99)
        x1 = random.uniform(0.05, 0.99-w)
        y1 = random.uniform(0.05, 0.99-h)
        return x1, y1, x1+w, y1+h

    def __getitem__(self, idx):
        img_idx = idx // self.masks_per_image
        img = Image.open(self.files[img_idx]).convert("RGB")
        img = self.tf(img)

        bbox = self._gen_bbox()
        C,H,W = img.shape
        mask = torch.zeros_like(img)
        x1 = int(bbox[0]*W); y1=int(bbox[1]*H)
        x2 = int(bbox[2]*W); y2=int(bbox[3]*H)
        mask[:, x1:x2, y1:y2] = 1.0

        masked_img = img * (1-mask)
        return masked_img, img, mask, torch.tensor(bbox), ""


# -------------------------
# Adapt UNet
# -------------------------
def adapt_unet_conv_in(unet, new_in):
    conv = unet.conv_in
    old_w = conv.weight.data
    out_ch, old_in, kH, kW = old_w.shape

    new_conv = nn.Conv2d(new_in, out_ch, conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         bias=(conv.bias is not None))

    with torch.no_grad():
        new_w = torch.zeros((out_ch, new_in, kH, kW), dtype=old_w.dtype)
        new_w[:, :old_in] = old_w
        mean_w = old_w.mean(dim=1, keepdim=True)
        for i in range(old_in, new_in):
            new_w[:, i:i+1] = mean_w

        new_conv.weight.copy_(new_w)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    unet.conv_in = new_conv
    print(f"[UNet] conv_in adapted {old_in} → {new_in}")
    return unet


# -------------------------
# Trainer
# -------------------------
class SD21InpaintTrainer_NoText:
    def __init__(self):
        self.accelerator = Accelerator(mixed_precision=MIXED_PRECISION)

        print("Loading SD2.1 UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(SD2_BASE, subfolder="unet")

        print("Loading SD2.1 VAE...")
        self.vae = AutoencoderKL.from_pretrained(SD2_BASE, subfolder="vae")

        print("Loading scheduler...")
        self.scheduler = DDPMScheduler.from_pretrained(SD2_BASE, subfolder="scheduler")

        print("Loading OpenCLIP Vision...")
        self.vision_proc = CLIPImageProcessor.from_pretrained(CLIP_VISION)
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_VISION)
        self.vision_encoder.requires_grad_(False)

        # adapt UNet input channels
        c = self.vae.config.latent_channels
        new_in = c + c + 1
        self.unet = adapt_unet_conv_in(self.unet, new_in)

        # freeze VAE
        self.vae.requires_grad_(False)

        # datasets
        self.train_loader = DataLoader(OutpaintDataset(TRAIN_ROOT), batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(OutpaintDataset(VAL_ROOT, masks_per_image=10), batch_size=VAL_BATCH_SIZE)

        # optimizer
        self.optimizer = torch.optim.AdamW([p for p in self.unet.parameters() if p.requires_grad], lr=LR)

        (
            self.unet,
            self.vae,
            self.vision_encoder,
            self.optimizer,
            self.train_loader,
            self.val_loader
        ) = self.accelerator.prepare(
            self.unet,
            self.vae,
            self.vision_encoder,
            self.optimizer,
            self.train_loader,
            self.val_loader
        )

        self.scaling = self.vae.config.scaling_factor

    # -------------------------
    def _mask_to_latent(self, mask, shape):
        B,_,H,W = mask.shape
        _,_,h,w = shape
        m = mask[:, :1]
        mlat = F.interpolate(m, (h,w), mode="nearest")
        return (mlat > LATENT_MASK_THRESHOLD).float()

    # -------------------------
    def _vision_embeds(self, masked):
        imgs = (masked + 1)*0.5
        imgs = imgs.clamp(0,1)
        proc = self.vision_proc(images=imgs, return_tensors="pt")
        pv = proc["pixel_values"].to(self.accelerator.device)
        with torch.no_grad():
            out = self.vision_encoder(pixel_values=pv)
        emb = out.image_embeds
        return emb.unsqueeze(1).expand(-1,64,-1)  # [B,64,dim]

    # -------------------------
    def train_step(self, batch):
        masked, target, mask, bbox, prompts = batch
        masked = masked.to(self.accelerator.device)
        target = target.to(self.accelerator.device)
        mask   = mask.to(self.accelerator.device)

        # encode target
        with torch.no_grad():
            t_lat = self.vae.encode(target).latent_dist.sample() * self.scaling

        m_lat = self._mask_to_latent(mask, t_lat.shape)
        noise = torch.randn_like(t_lat)
        ts = torch.randint(0, self.scheduler.num_train_timesteps, (t_lat.size(0),), device=self.accelerator.device)

        noisy_masked = self.scheduler.add_noise(t_lat*m_lat, noise*m_lat, ts)
        noisy_lat = t_lat*(1-m_lat) + noisy_masked

        with torch.no_grad():
            masked_lat = self.vae.encode(masked).latent_dist.sample() * self.scaling

        unet_input = torch.cat([noisy_lat, masked_lat, m_lat], dim=1)

        img_emb = self._vision_embeds(masked)
        enc_hid = img_emb  # ONLY VISION EMBEDDING (NO TEXT)

        out = self.unet(unet_input, ts, encoder_hidden_states=enc_hid).sample
        loss = F.mse_loss(out*m_lat, noise*m_lat)
        return loss

    # -------------------------
    def validate(self):
        self.unet.eval()
        tot,n = 0,0
        with torch.no_grad():
            for b in self.val_loader:
                loss = self.train_step(b)
                tot+=loss.item(); n+=1
        self.unet.train()
        print("VAL:", tot/n)
        return tot/n

    # -------------------------
    def train(self):
        best = 1e9
        patience = 0
        for ep in range(EPOCHS):
            print("Epoch", ep+1)
            losses=[]
            for batch in tqdm(self.train_loader):
                with self.accelerator.accumulate(self.unet):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                losses.append(loss.item())

            print("Train:", np.mean(losses))
            val = self.validate()

            if val < best:
                best = val
                patience=0
                self.accelerator.save_state(os.path.join(SAVE_DIR))
            else:
                patience+=1
                if patience>=PATIENCE:
                    print("Early stop.")
                    return


# -------------------------
# RUN
# -------------------------
trainer = SD21InpaintTrainer_NoText()
trainer.unet.enable_gradient_checkpointing()
trainer.accelerator.load_state("drive/MyDrive/sd2")
trainer.train()
