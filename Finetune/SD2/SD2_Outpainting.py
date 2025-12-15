# sd2_outpaint_infer.py
import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

import safetensors.torch as safetorch

# ---------- config ----------
SD2_BASE = "Manojb/stable-diffusion-2-1-base"
CLIP_VISION = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CHECKPOINT_DIR = "your_path"
OUTINFER_ROOT = "your_path"
# ----------------------------

def find_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors found in {CHECKPOINT_DIR}")
    main = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    if os.path.exists(main):
        return main
    return sorted(files, key=os.path.getmtime)[-1]

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

class OutpaintEngine:
    def __init__(self, device="cuda"):
        self.device = device
        self.accelerator = Accelerator(mixed_precision="fp16")

        # Load SD2 components
        print("Loading SD2 UNet / VAE / Scheduler...")
        self.unet = UNet2DConditionModel.from_pretrained(SD2_BASE, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(SD2_BASE, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(SD2_BASE, subfolder="scheduler")

        # CLIP vision encoder for condition
        print("Loading CLIP vision encoder...")
        self.vision_proc = CLIPImageProcessor.from_pretrained(CLIP_VISION)
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_VISION)
        self.vision_encoder.eval()
        self.vision_encoder.requires_grad_(False)

        # Expand UNet conv_in to match training: c + c + 1
        latent_c = self.vae.config.latent_channels
        new_in = latent_c + latent_c + 1
        self.unet = adapt_unet_conv_in(self.unet, new_in)

        # Prepare with accelerator
        (self.unet, self.vae, self.vision_encoder) = self.accelerator.prepare(
            self.unet, self.vae, self.vision_encoder
        )

        # load finetuned checkpoint (safetensors)
        ckpt = find_checkpoint()
        print("Loading finetuned weights from", ckpt)
        state = safetorch.load_file(ckpt)
        # try to map keys directly; allow non-strict
        missing, unexpected = self.unet.load_state_dict(state, strict=True)
        print("UNet load_state_dict missing keys:", len(missing), "unexpected keys:", len(unexpected))

        # mode & device
        self.vae.eval().to(self.accelerator.device)
        self.unet.eval().to(self.accelerator.device)
        self.vision_encoder.to(self.accelerator.device)

        # directions and transform
        self.directions = ['right', 'left', 'down', 'up']
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def _extract_old_region(self, image_tensor, direction, crop_ratio):
        b, c, h, w = image_tensor.shape
        masked = image_tensor.clone()
        if direction in [0, 1]:
            crop_w = int(w * crop_ratio)
            if direction == 0:
                masked = masked[..., :, :, :crop_w]
            else:
                masked = masked[..., :, :, w - crop_w:]
        else:
            crop_h = int(h * crop_ratio)
            if direction == 2:
                masked = masked[..., :crop_h, :]
            else:
                masked = masked[..., h - crop_h:, :]
        return masked

    def _create_latent_mask(self, bbox, latent_shape):
        b, _, lh, lw = latent_shape
        masks = []
        # bbox: tensor of shape [N,4] normalized x1,y1,x2,y2
        for coords in bbox:
            x1 = coords[0] * lw
            y1 = coords[1] * lh
            x2 = coords[2] * lw
            y2 = coords[3] * lh
            # create meshgrid with correct order
            xx, yy = torch.meshgrid(
                torch.arange(lh, device=self.accelerator.device),
                torch.arange(lw, device=self.accelerator.device),
                indexing='ij'
            )
            mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
            masks.append(mask)
        return torch.stack(masks).unsqueeze(1)  # [N,1,lh,lw]

    def _vision_embeds_from_img(self, img_tensor):
        # img_tensor in range [-1,1], shape [B,3,H,W]
        imgs = (img_tensor + 1.0) * 0.5
        imgs = imgs.clamp(0,1)
        # prepare pixel_values via processor for batch
        pil_imgs = [Image.fromarray((imgs[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)) for i in range(imgs.shape[0])]
        proc = self.vision_proc(images=pil_imgs, return_tensors="pt")
        pv = proc["pixel_values"].to(self.accelerator.device)
        with torch.no_grad():
            out = self.vision_encoder(pixel_values=pv)
        emb = out.image_embeds  # [B, dim]
        return emb.unsqueeze(1).expand(-1, 64, -1)  # [B,64,dim]

    def generate_iterative(self, image_tensor, steps=200, crop_ratio=0.95, iterations=10, direction="right", name="default"):
        direction_idx = self.directions.index(direction)
        current_image = image_tensor.clone()
        b, c, h, w = current_image.shape
        lh, lw = 64, 64

        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        crop_lw, crop_lh = int(lw * crop_ratio), int(lh * crop_ratio)
        mask_size = w - crop_w if direction in ["left", "right"] else h - crop_h
        latent_mask_size = lw - crop_lw if direction in ["left", "right"] else lh - crop_lh

        preserved_region = self._extract_old_region(current_image, direction_idx, crop_ratio)
        with torch.no_grad():
            vae_decoded = self.vae.decode(self.vae.encode(current_image).latent_dist.sample()).sample
            plt.figure(figsize=(6,6))
            st_vis = (vae_decoded[0].permute(1,2,0).cpu().numpy()*0.5 + 0.5).clip(0,1)
            plt.imshow(st_vis)
            plt.axis("off")
            plt.title("origin")
            plt.show()

        stitched = preserved_region
        current_latent = None

        if direction == "right":
            bbox = torch.tensor([[0.0, crop_ratio, 1.0, 1.0]], device=self.accelerator.device)
        elif direction == "left":
            bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0 - crop_ratio]], device=self.accelerator.device)
        elif direction == "down":
            bbox = torch.tensor([[crop_ratio, 0.0, 1.0, 1.0]], device=self.accelerator.device)
        else:
            bbox = torch.tensor([[0.0, 0.0, 1.0 - crop_ratio, 1.0]], device=self.accelerator.device)

        for i in range(iterations):
            print(f"[{i+1}/{iterations}] Expanding → {direction.upper()}")
            if i == 0:
                mask = torch.zeros_like(current_image)
                x1 = int(bbox[0][0] * w)
                y1 = int(bbox[0][1] * h)
                x2 = int(bbox[0][2] * w)
                y2 = int(bbox[0][3] * h)
                # note: original code used mask[:, :, x1:x2, y1:y2]
                mask[:, :, x1:x2, y1:y2] = 1.0
                masked_img = current_image * (1 - mask)
                with torch.no_grad():
                    masked_latents = self.vae.encode(masked_img).latent_dist.sample()
                    masked_latents = masked_latents * self.vae.config.scaling_factor
                    vae_decoded = self.vae.decode(masked_latents / self.vae.config.scaling_factor).sample
                    plt.figure(figsize=(6,6))
                    st_vis = (vae_decoded[0].permute(1,2,0).cpu().numpy()*0.5 + 0.5).clip(0,1)
                    plt.imshow(st_vis)
                    plt.axis("off")
                    plt.title("masked")
                    plt.show()
                # compute vision condition from masked image
                condition = self._vision_embeds_from_img(masked_img)
            else:
                masked_latents = current_latent
                # for consistency, re-use previous condition (vision condition derived from preserved masked context)
                # condition already set in first iter

            latent_mask = self._create_latent_mask(bbox, masked_latents.shape)  # [1,1,lh,lw]

            # add noise to masked region
            noise = torch.randn_like(masked_latents, device=self.accelerator.device)
            # add_noise expects timesteps scalar/tensor; keep same pattern as original (torch.tensor(steps))
            noisy_part = self.noise_scheduler.add_noise(masked_latents * latent_mask, noise * latent_mask,
                                                       torch.tensor(steps, device=self.accelerator.device))
            noisy_latents = masked_latents * (1 - latent_mask) + noisy_part * latent_mask

            # denoising
            self.noise_scheduler.set_timesteps(steps)
            latent_input = noisy_latents

            # UNet expects latent_input shaped and encoder_hidden_states = condition
            for t in self.noise_scheduler.timesteps:
                latent_input = latent_input * latent_mask + masked_latents * (1 - latent_mask)
                unet_input = torch.cat([
                  latent_input,     # noisy_lat (B,4,64,64)
                  masked_latents,   # masked_lat (B,4,64,64)
                  latent_mask       # mask (B,1,64,64)
                ], dim=1)
                with torch.no_grad():
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=condition).sample
                latent_input = self.noise_scheduler.step(noise_pred, t, latent_input).prev_sample

            # decode
            with torch.no_grad():
                generated_latent = masked_latents * (1 - latent_mask) + latent_input * latent_mask
                generated_img = self.vae.decode(generated_latent / self.vae.config.scaling_factor).sample

            # visualize / save a preview as in original
            plt.figure(figsize=(6,6))
            vis = (generated_img[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5).clip(0,1)
            plt.imshow(vis)
            plt.title("generated")
            plt.axis("off")
            plt.show()

            new_patch = self._extract_new_region(generated_img, direction_idx, mask_size)
            stitched = self._stitch_image(stitched, new_patch, direction_idx)

            # save current_latent/current_image
            latent_save_path = f"{OUTINFER_ROOT}/{name}/{i:02d}_latent.pt"
            image_save_path = f"{OUTINFER_ROOT}/{name}/{i:02d}_image.png"
            os.makedirs(os.path.dirname(latent_save_path), exist_ok=True)
            # save latent
            torch.save(generated_latent.cpu(), latent_save_path)
            current_img = (generated_img[0].clamp(-1,1) * 0.5 + 0.5).cpu()
            Image.fromarray((current_img.permute(1,2,0).numpy()*255).astype(np.uint8)).save(image_save_path)

            # prepare for next iter: cyclic shift both image and latent
            current_image = self._cyclic_shift(generated_img, direction_idx, mask_size)
            current_latent = self._cyclic_latent_shift(generated_latent, direction_idx, latent_mask_size)

            # display stitched
            plt.figure(figsize=(6,6))
            st_vis = (stitched[0].permute(1,2,0).cpu().numpy()*0.5 + 0.5).clip(0,1)
            plt.imshow(st_vis)
            plt.axis("off")
            plt.title("stitched")
            plt.show()

        return ((stitched[0].permute(1,2,0).cpu().numpy()*0.5 + 0.5).clip(0,1) * 255).astype(np.uint8)

    def _extract_new_region(self, generated, direction, mask_size):
        if direction == 0:
            return generated[..., :, :, -mask_size:]
        elif direction == 1:
            return generated[..., :, :, :mask_size]
        elif direction == 2:
            return generated[..., :, -mask_size:, :]
        elif direction == 3:
            return generated[..., :, :mask_size, :]

    def _stitch_image(self, combined, generated_patch, direction):
        if direction == 0:
            combined = torch.cat([combined, generated_patch], dim=-1)
        elif direction == 1:
            combined = torch.cat([generated_patch, combined], dim=-1)
        elif direction == 2:
            combined = torch.cat([combined, generated_patch], dim=-2)
        elif direction == 3:
            combined = torch.cat([generated_patch, combined], dim=-2)
        return combined

    def _cyclic_shift(self, generated, direction, mask_size):
        if direction == 0:
            return torch.cat([generated[..., :, :, mask_size:], generated[..., :, :, :mask_size]], dim=-1)
        elif direction == 1:
            return torch.cat([generated[..., :, :, -mask_size:], generated[..., :, :, :-mask_size]], dim=-1)
        elif direction == 2:
            return torch.cat([generated[..., :, mask_size:, :], generated[..., :, :mask_size, :]], dim=-2)
        elif direction == 3:
            return torch.cat([generated[..., :, -mask_size:, :], generated[..., :, :-mask_size, :]], dim=-2)

    def _cyclic_latent_shift(self, generated, direction, mask_size):
        # same logic for latent tensors
        if direction == 0:
            return torch.cat([generated[..., :, :, mask_size:], torch.zeros_like(generated[..., :, :, :mask_size])], dim=-1)
        elif direction == 1:
            return torch.cat([torch.zeros_like(generated[..., :, :, -mask_size:]), generated[..., :, :, :-mask_size]], dim=-1)
        elif direction == 2:
            return torch.cat([generated[..., :, mask_size:, :], torch.zeros_like(generated[..., :, :mask_size, :])], dim=-2)
        elif direction == 3:
            return torch.cat([torch.zeros_like(generated[..., :, -mask_size:, :]), generated[..., :, :-mask_size, :]], dim=-2)

# ---------------- main  ----------------
if __name__ == "__main__":
    root_dir = "drive/MyDrive/outtest"
    img_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]

    outpaint_engine = OutpaintEngine()
    for i in img_files:
        for j in range("your_attempts_times"):
            img = Image.open(i).convert("RGB")
            img_t = outpaint_engine.transform(img).unsqueeze(0).to(outpaint_engine.accelerator.device)
            stitched_image = outpaint_engine.generate_iterative(img_t, steps=200, crop_ratio=0.95, iterations=6, direction="up", name=f"{os.path.basename(i)}_{j}")
            os.makedirs("outpath", exist_ok=True)
            Image.fromarray(stitched_image).save(f"outpath/{os.path.basename(i)}+{j}.png")
