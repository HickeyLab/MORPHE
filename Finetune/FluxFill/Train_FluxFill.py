import os
import math
import random
import copy
from safetensors.torch import load_file
import itertools
import warnings
from pathlib import Path
from diffusers.training_utils import cast_training_params
from peft.utils import get_peft_model_state_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from diffusers import FluxFillPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from peft import LoraConfig
from transformers import CLIPTokenizer, T5TokenizerFast

# -------------------------
# CONFIG
# -------------------------
HF_REPO = "black-forest-labs/FLUX.1-Fill-dev"
TRAIN_ROOT = "your_path"
MASK_ROOT = None

IMG_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 10
LR = 5e-6
SEED = 42
MIXED_PRECISION = "fp16"

LORA_RANK = 4
SAVE_DIR = "your_path"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_WORKERS = 1

# -------------------------
# utils
# -------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()


# -------------------------
# Dataset (simple)
# -------------------------
class RandomMaskDataset(Dataset):
    def __init__(self, img_dir, size=512, masks_per_image=50):
        self.size = size
        self.masks_per_image = masks_per_image
        self.img_paths = sorted([os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg'))])

        if len(self.img_paths) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()

    def _gen_bbox(self):
        if random.random() < 0.5:
            ar = random.uniform(1, 33)
        else:
            ar = random.uniform(0.03, 1)

        area = random.uniform(0.1,0.33)**2
        w = min(math.sqrt(area*ar), 0.99)
        h = min(math.sqrt(area/ar), 0.99)
        x1 = random.uniform(0.01, 0.99-w)
        y1 = random.uniform(0.01, 0.99-h)
        return x1, y1, x1+w, y1+h

    def __len__(self):
        return max(1, len(self.img_paths) * self.masks_per_image)

    def __getitem__(self, idx):
        img_idx = idx // self.masks_per_image
        p = self.img_paths[img_idx % len(self.img_paths)]
        img = Image.open(p).convert("RGB")
        #img = self.resize(img)
        img = self.to_tensor(img)        # [0,1]
        img = img * 2.0 - 1.0         # [-1,1]

        H, W = self.size, self.size
        x1, y1, x2, y2 = self._gen_bbox()
        x1p, y1p = int(x1 * W), int(y1 * H)
        x2p, y2p = int(x2 * W), int(y2 * H)
        mask = torch.zeros(1, H, W)
        if x2p <= x1p: x2p = min(x1p + 1, W)
        if y2p <= y1p: y2p = min(y1p + 1, H)
        mask[:, y1p:y2p, x1p:x2p] = 1.0

        masked_img = img * (1 - mask)

        return img, mask, masked_img


# -------------------------
# Trainer (pack-style)
# -------------------------
class FluxTrainerPack:
    def __init__(self, resume_lora_path=None):
        self.acc = Accelerator(mixed_precision=MIXED_PRECISION)
        self.device = self.acc.device
        self.text_tokens = 1
        self.pooled_dim = 768

        print("Loading FluxFillPipeline...")
        self.pipe = FluxFillPipeline.from_pretrained(
            HF_REPO,
            torch_dtype=torch.float16
        )

        self.vae = self.pipe.vae.to(self.device)
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler

        # Move scheduler tensors
        for k, v in list(self.scheduler.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self.scheduler, k, v.to(self.device))

        # freeze transformer backbone + vae
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)

        # ---- LoRA config ----
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
        ]

        self.lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            init_lora_weights="gaussian",
            target_modules=target_modules
        )

        # Inject LoRA
        self.transformer.add_adapter(self.lora_cfg)
        # ---- Resume training (load existing LoRA) ----
        if resume_lora_path is not None:
            print(f"[Resume LoRA] Loading: {resume_lora_path}")
            # load LoRA safely using pipeline loader
            self.pipe.load_lora_weights(resume_lora_path)
            #state = load_file(os.path.join(resume_lora_path, "pytorch_lora_weights.safetensors"))
            #missing, unexpected = self.transformer.load_state_dict(state, strict=False)
            # sync the transformer with updated weights
            self.transformer = self.pipe.transformer
        else:

            print("[Resume LoRA] None, training from scratch.")

        # ---- Upcast ONLY LoRA parameters to fp32 ----
        cast_training_params([self.transformer], dtype=torch.float32)

        # dataset
        self.ds = RandomMaskDataset(TRAIN_ROOT, size=IMG_SIZE)
        self.dl = DataLoader(self.ds,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

        # optimizer (only LoRA params)
        trainable = [p for p in self.transformer.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable, lr=LR)

        # prepare models
        (self.transformer, self.opt, self.dl) = self.acc.prepare(
            self.transformer, self.opt, self.dl
        )

        self.vae.eval()

    def train(self, epochs=EPOCHS):
        transformer = self.transformer
        scheduler = self.scheduler
        vae = self.vae
        pipe = self.pipe

        for ep in range(epochs):
            transformer.train()
            pbar = tqdm(self.dl, desc=f"Epoch {ep+1}/{epochs}")
            losses = []

            for batch in pbar:
                loss = self._train_step(batch, transformer, vae, scheduler, pipe)
                losses.append(loss.item())
                pbar.set_postfix({"loss": sum(losses)/len(losses)})

            # Plot loss curve per epoch
            if self.acc.is_main_process:
                plt.figure(figsize=(8,5))
                plt.plot(losses)
                plt.title(f"Loss - Epoch {ep+1}")
                plt.savefig(os.path.join(SAVE_DIR, f"loss_epoch_{ep+1}.png"))
                plt.close()

                # ------- SAVE REAL LORA -------
                print("Saving LoRA...")
                transformer_lora_layers_to_save = get_peft_model_state_dict(transformer)
                self.pipe.save_lora_weights(os.path.join(SAVE_DIR, f"lora"), transformer_lora_layers=transformer_lora_layers_to_save)
                print(f"[Saved] LoRA at: {SAVE_DIR}/lora")

    def _train_step(self, batch, transformer, vae, scheduler, pipe):
        img, mask, masked_img = batch
        img = img.to(self.device, dtype=vae.dtype)
        mask = mask.to(self.device, dtype=vae.dtype)
        orig_mask = mask
        masked_img = masked_img.to(self.device, dtype=vae.dtype)
        vae_scale = 2 ** (len(vae.config.block_out_channels)-1)

        B = img.shape[0]

        # 1) VAE encode
        with torch.no_grad():
            lat = vae.encode(img).latent_dist.sample()
            lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
            masked_lat = vae.encode(masked_img).latent_dist.sample()
            masked_lat = (masked_lat - vae.config.shift_factor) * vae.config.scaling_factor

        _, C, Hlat, Wlat = lat.shape

        # 2) flow-matching noise & timestep
        T = getattr(scheduler.config, "num_train_timesteps", None)
        if T is None:
            T = len(getattr(scheduler, "timesteps", []))
        # sample indices uniformly (you can replace with official density sampling)
        t_idx = torch.randint(0, T, (B,), device=self.device)
        timesteps = scheduler.timesteps[t_idx].to(device=self.device)
        # get sigma from scheduler.sigmas
        sigmas = scheduler.sigmas[t_idx].view(B,1,1,1)
        noise = torch.randn_like(lat)

        noisy_masked_lat = (1 - sigmas) * masked_lat + sigmas * noise

        # mask downsampled to latents
        mask_small = F.interpolate(mask, size=(Hlat, Wlat), mode="nearest")
        mask_bc = mask_small.expand(_, C, Hlat, Wlat)

        noisy_input_lat = lat*(1-mask_bc) + noisy_masked_lat*mask_bc  # [B,C,Hlat,Wlat]

        # 3) pack latents using pipeline helper (official layout)
        packed_noisy = pipe._pack_latents(
            noisy_input_lat, batch_size=B, num_channels_latents=C, height=Hlat, width=Wlat
        )  # [B, seq_packed, packed_dim]
        packed_masked = pipe._pack_latents(
            masked_lat, batch_size=B, num_channels_latents=C, height=Hlat, width=Wlat
        )

        # prepare mask to pack: expand to channels to be packable
        # here we expand mask_small to C channels
        mask = mask.reshape(-1, 1, 512 // 8, 512 // 8)

        mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
        mask = mask.view(lat.shape[0], lat.shape[2], vae_scale, lat.shape[3], vae_scale)  # batch_size, height, 8, width, 8
        mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
        mask = mask.reshape(
                    lat.shape[0], vae_scale * vae_scale, lat.shape[2], lat.shape[3]
                )
        packed_mask = pipe._pack_latents(
            mask, batch_size=B, num_channels_latents=vae_scale*vae_scale, height=Hlat, width=Wlat
        )

        # concat masked image packed with mask along token dim (last dim)
        # masked_image_latents already packed -> concat mask on channel axis (dim=2)
        masked_image_latents = torch.cat((packed_masked, packed_mask), dim=2)

        # official pattern: concat packed_noisy and masked_image_latents along channel axis
        transformer_input = torch.cat((packed_noisy, masked_image_latents), dim=2)  # [B, seq_packed, total_dim]

        # 4) prepare encoder (text) embeddings: if you have no text, use zero prompt_embeds
        # create dummy prompt_embeds: B x text_tokens x joint_dim
        prompt_embeds = torch.zeros(B, self.text_tokens, 4096, device=self.device, dtype=transformer_input.dtype)
        pooled_prompt_embeds = torch.zeros(B, self.pooled_dim, device=self.device, dtype=transformer_input.dtype)
        text_ids = torch.zeros(self.text_tokens, 3, dtype=torch.long, device=self.device)

        # 5) prepare img_ids (official helper)
        latent_image_ids = pipe._prepare_latent_image_ids(
            batch_size=B,
            height=Hlat//2,  # official used Hlat//2 in some places; use pipeline helper pattern - here keep Hlat
            width=Wlat//2,
            device=self.device,
            dtype=torch.long,
        )
        # note: earlier official code used Hlat//2 etc; pipe._prepare_latent_image_ids must match pack layout
        # safer to call with Hlat and Wlat that match pipeline expectations; if mismatch, adjust to Hlat//2

        # guidance
        guidance = None
        if getattr(torch, "is_tensor", None):
            pass
        if getattr(transformer.config, "guidance_embeds", False):
            guidance = torch.zeros(B, device=self.device, dtype=transformer_input.dtype)


        # convert timesteps for transformer (scale)
        timestep_argument = timesteps / 1000.0

        # 6) forward through tr ansformer - official passes packed tokens directly
        # ensure dtype: transformer may expect float32/float16; pack value dtype will be float16 due to vae dtype
        # The transformer's forward accepts hidden_states in packed format.
        #print(transformer_input.shape)
        #print(prompt_embeds.shape)
        #print(pooled_prompt_embeds.shape)
        model_pred = transformer(
            hidden_states=transformer_input,
            timestep=timestep_argument,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        model_pred = FluxFillPipeline._unpack_latents(
                    model_pred,
                    height=lat.shape[2] * vae_scale,
                    width=lat.shape[3] * vae_scale,
                    vae_scale_factor=vae_scale,
                )

        # 7) flow-matching target
        target = (noise - lat).to(model_pred.dtype)
      
        loss = F.mse_loss((model_pred * mask_bc).float(), (target * mask_bc).float())
        #loss = F.mse_loss(model_pred, target)

        # backward & step
        self.acc.backward(loss)
        self.opt.step()
        self.opt.zero_grad()

        return loss

# -------------------------
# run
# -------------------------
if __name__ == "__main__":
    trainer = FluxTrainerPack("drive/MyDrive/fluxfill_lora/lora")
    trainer.train()
