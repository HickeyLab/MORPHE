import os
import glob
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torchvision import transforms
from diffusers import FluxFillPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
import safetensors.torch as safetorch

# ---------------- CONFIG ----------------
HF_REPO = "black-forest-labs/FLUX.1-Fill-dev"   # same as training
LORA_PATH = "drive/MyDrive/fluxfill_lora/lora"  # directory saved by pipe.save_lora_weights(...)
OUTINFER_ROOT = "drive/MyDrive/fluxfill_outinfer"
IMG_ROOT = "drive/MyDrive/outtest"  # images to outpaint
DEVICE = "cuda"
# inference defaults
STEPS = 50
IMG_SIZE = 512
# ----------------------------------------

os.makedirs(OUTINFER_ROOT, exist_ok=True)

# helper: image <-> tensor transform matching training
transform = transforms.Compose([
    #transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)   # -> [-1,1]
])

def tensor_to_pil(img_tensor):
    # img_tensor in [B,3,H,W] in [0,1] or [-1,1] depending - here expect [-1,1] or [0,1]
    t = img_tensor
    if t.min() < 0:
        t = (t.clamp(-1,1)*0.5 + 0.5)
    arr = (t[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return Image.fromarray(arr)

class OutpaintEngineFlux:
    def __init__(self, device=DEVICE, lora_path: str = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # load pipeline (same repo as training)
        print("Loading FluxFillPipeline...")
        self.pipe = FluxFillPipeline.from_pretrained(HF_REPO, torch_dtype=torch.float16)
        # Pull pieces
        self.vae = self.pipe.vae.to(self.device)
        self.transformer = self.pipe.transformer.to(self.device)
        self.scheduler = self.pipe.scheduler
        for k, v in list(self.scheduler.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self.scheduler, k, v.to(self.device))

        # move scheduler tensors to device if any
        for k, v in list(self.scheduler.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self.scheduler, k, v.to(self.device))

        # load LoRA if provided
        if lora_path is not None and os.path.exists(lora_path):
            print(f"Loading LoRA from: {lora_path}")
            self.pipe.load_lora_weights(lora_path)
            # sync transformer in case pipe replaced adapters
            self.transformer = self.pipe.transformer

        # make sure vae/transformer on device and in eval
        self.vae.eval().to(self.device)
        self.transformer.eval().to(self.device)

        # directions and helper
        self.directions = ['right', 'left', 'down', 'up']
        self.transform = transform

    def _extract_old_region(self, image_tensor, direction, crop_ratio):
        b,c,h,w = image_tensor.shape
        masked = image_tensor.clone()
        if direction in [0,1]:
            crop_w = int(w * crop_ratio)
            if direction == 0:
                masked = masked[..., :, :, :crop_w]
            else:
                masked = masked[..., :, :, w-crop_w:]
        else:
            crop_h = int(h * crop_ratio)
            if direction == 2:
                masked = masked[..., :crop_h, :]
            else:
                masked = masked[..., h-crop_h:, :]
        return masked

    def _create_bbox_for_direction(self, direction, crop_ratio):
        if direction == "right":
            return torch.tensor([[0.0, crop_ratio, 1.0, 1.0]], device=self.device)
        elif direction == "left":
            return torch.tensor([[0.0, 0.0, 1.0, 1.0 - crop_ratio]], device=self.device)
        elif direction == "down":
            return torch.tensor([[crop_ratio, 0.0, 1.0, 1.0]], device=self.device)
        else:
            return torch.tensor([[0.0, 0.0, 1.0 - crop_ratio, 1.0]], device=self.device)

    def _create_mask_from_bbox(self, bbox, H, W):
        x1 = int(bbox[0,0].item() * W); y1 = int(bbox[0,1].item() * H)
        x2 = int(bbox[0,2].item() * W); y2 = int(bbox[0,3].item() * H)
        m = torch.zeros(1, H, W, device=self.device)
        m[:, x1:x2, y1:y2] = 1.0
        return m  # [1,H,W]

    def _cyclic_latent_shift(self, latent, direction_idx, latent_mask_size):
        # latent: [B, C, Hlat, Wlat]
        if direction_idx == 0:  # right: shift left by mask_size
            return torch.cat([latent[..., :, :, latent_mask_size:], torch.zeros_like(latent[..., :, :, :latent_mask_size])], dim=-1)
        elif direction_idx == 1:  # left: shift right
            return torch.cat([torch.zeros_like(latent[..., :, :, -latent_mask_size:]), latent[..., :, :, :-latent_mask_size]], dim=-1)
        elif direction_idx == 2:  # down: shift up
            return torch.cat([latent[..., :, latent_mask_size:, :], torch.zeros_like(latent[..., :, :latent_mask_size, :])], dim=-2)
        else:  # up: shift down
            return torch.cat([torch.zeros_like(latent[..., :, -latent_mask_size:, :]), latent[..., :, :-latent_mask_size, :]], dim=-2)

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

    @torch.no_grad()
    def generate_iterative(self, image_tensor, steps=STEPS, crop_ratio=0.95, iterations=8, direction="right", name="default"):
        """
        Latent-level iterative outpainting:
         - encode once to get lat / masked_lat
         - iterate: denoise masked region in latent space using transformer + scheduler
         - update curr_latents in-place and cyclic shift for next iteration
        """
        assert image_tensor.shape[0] == 1, "Only B=1 supported in this helper"
        direction_idx = self.directions.index(direction)
        current_image = image_tensor.clone().to(self.device).half()
        b,c,Hpix,Wpix = current_image.shape

        # preserved region (for final stitch visualization)
        preserved_region = self._extract_old_region(current_image, direction_idx, crop_ratio)
        stitched = preserved_region

        # bbox (normalized)
        bbox = self._create_bbox_for_direction(direction, crop_ratio)

        # --- initial encode ONCE ---
        with torch.no_grad():
            lat = self.vae.encode(current_image).latent_dist.sample()   # [1,C,Hlat,Wlat]
            lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            # create initial masked image and encode masked_lat once (masked context)
            mask_pix = self._create_mask_from_bbox(bbox, Hpix, Wpix).to(dtype=torch.float16)  # [1,H,W]
            masked_img = current_image * (1 - mask_pix.unsqueeze(1)).to(dtype=torch.float16)
            masked_lat = self.vae.encode(masked_img).latent_dist.sample()
            masked_lat = (masked_lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        B, C, Hlat, Wlat = lat.shape
        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # latent mask (downsampled boolean mask in latent grid)
        mask_small = F.interpolate(mask_pix.unsqueeze(0), size=(Hlat, Wlat), mode="nearest")  # [1,1,Hlat,Wlat]
        latent_mask = mask_small  # [1,1,Hlat,Wlat]
        latent_mask_bc = latent_mask.expand(-1, C, -1, -1)  # [1,C,Hlat,Wlat]
          # prepare the packed mask token as in training (pack mask channel)
        mask_view = mask_pix.reshape(-1, 1, Hpix // 8, Wpix // 8)
        mask_view = mask_view[:, 0, :, :].view(B, Hlat, vae_scale, Wlat, vae_scale)
        mask_view = mask_view.permute(0, 2, 4, 1, 3)
        mask_view = mask_view.reshape(1, vae_scale * vae_scale, Hlat, Wlat)
        packed_mask = self.pipe._pack_latents(
                  mask_view.to(self.device), batch_size=1,
                  num_channels_latents=vae_scale * vae_scale,
                  height=Hlat, width=Wlat
                )

        # compute mask sizes for cyclic shift (in latent coords)
        # mask width/height in pixels for latent grid:
        mask_h_lat = int(mask_small.shape[2])
        mask_w_lat = int(mask_small.shape[3])
        crop_lw, crop_lh = int(mask_w_lat * crop_ratio), int(mask_h_lat * crop_ratio)
        latent_mask_size = mask_w_lat - crop_lw if direction_idx in [0,1] else mask_h_lat - crop_lh

        # initialize curr_latents: we keep the whole-image lat and will update masked region in-place
        curr_latents = None
        guidance = None
        if getattr(torch, "is_tensor", None):
            pass
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.zeros(1, device=self.device, dtype=self.transformer.dtype)

        # iterate outpainting rounds
        for it in range(iterations):
            print(f"[Iter {it+1}/{iterations}] Outpainting → {direction.upper()}")

            # For first iteration: masked_lat is from masked_img; for later iterations masked_lat is previous generated_latent
            if it == 0:
                masked_region_lat = masked_lat.clone()
            else:
                # masked_region_lat we set equal to the masked area extracted from curr_latents
                # (this maintains context around the masked area)
                masked_region_lat = curr_latents * (1 - latent_mask_bc)

            #print(masked_region_lat.shape)

            # Build packed versions used by transformer (match training packing)
            packed_masked = self.pipe._pack_latents(
                masked_region_lat, batch_size=B, num_channels_latents=C, height=Hlat, width=Wlat
            )
            packed_noisy = None  # will be re-built per timestep from curr_latents

            # masked image packed (masked_latents + packed_mask) same dim as training
            masked_image_latents = torch.cat((packed_masked, packed_mask), dim=2)

            # prepare prompt / pooled embeddings (zeros when no text)
            prompt_embeds = torch.zeros(1, 1, getattr(self.transformer.config, "joint_attention_dim", 4096),
                                        device=self.device, dtype=masked_image_latents.dtype)
            pooled_prompt_embeds = torch.zeros(1, getattr(self.transformer.config, "pooled_projection_dim", 768),
                                        device=self.device, dtype=masked_image_latents.dtype)
            text_ids = torch.zeros(1, 3, dtype=torch.long, device=self.device)

            latent_image_ids = self.pipe._prepare_latent_image_ids(
                    batch_size=1,
                    height=Hlat // 2,
                    width=Wlat // 2,
                    device=self.device,
                    dtype=torch.long,
                )

            # scheduler timesteps setup
            mu = getattr(self.scheduler.config, "mu", 0.5)
            self.scheduler.set_timesteps(steps, mu=mu)
            self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
            # prepare curr_latents copy that we will update through timesteps
            # start from full-image lat but we only replace masked region during step updates
            working_latents = masked_region_lat.clone()
            sigmas = self.scheduler.sigmas[torch.tensor(steps-1, dtype=torch.int)].view(B,1,1,1).to(self.device)
            noise = torch.randn_like(working_latents).to(self.device)
            noisy_part = (1 - sigmas) * working_latents + sigmas * noise
            working_latents = (1 - latent_mask_bc).to(self.device) * working_latents.to(self.device) + latent_mask_bc.to(self.device) * noisy_part.to(self.device)
            #print(working_latents.shape)

            # per-timestep denoising loop (like training)
            for t in range(steps):

                packed_noisy = self.pipe._pack_latents(
                    working_latents, batch_size=1, num_channels_latents=C, height=Hlat, width=Wlat
                ).half()
                # transformer input concatenates packed_noisy + masked_image_latents
                transformer_input = torch.cat((packed_noisy, masked_image_latents), dim=2).half()
                #print(self.scheduler.timesteps[t].to(device=self.device))
                timestep = self.scheduler.timesteps[torch.tensor([t],device=self.device,dtype=torch.int)].to(device=self.device) / 1000.0

                # run transformer to predict packed output
                model_pred = self.transformer(
                    hidden_states=transformer_input.to(self.device).half(),
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # unpack to pixel-latent prediction
                pred_unpacked = self.pipe._unpack_latents(
                    model_pred,
                    height=Hlat * vae_scale,
                    width=Wlat * vae_scale,
                    vae_scale_factor=vae_scale
                )  # [1, C, Hlat, Wlat]

                # scheduler step: update working_latents.
                step_out = self.scheduler.step(pred_unpacked, self.scheduler.timesteps[torch.tensor([t],device=self.device,dtype=torch.int)].to(device=self.device), working_latents)
                working_latents = step_out.prev_sample

                # keep only masked region updated; keep unmasked region from original lat
                working_latents = working_latents * latent_mask_bc + masked_region_lat * (1 - latent_mask_bc)

            # end timesteps loop

            # after finishing timesteps, build generated_latent by merging working_latents into lat on mask
            generated_latent =  working_latents * latent_mask_bc + masked_region_lat * (1 - latent_mask_bc)

            # decode generated_latent for visualization / saving
            with torch.no_grad():
                decoded = self.vae.decode(generated_latent / self.vae.config.scaling_factor).sample  # [-1,1]

            # extract new patch according to direction & stitch
            new_patch = self._extract_new_region(decoded, direction_idx, mask_w_lat if direction_idx in [0,1] else mask_h_lat)
            stitched = self._stitch_image(stitched, new_patch, direction_idx)

            # save latent + image
            save_dir = os.path.join(OUTINFER_ROOT, name)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(generated_latent.cpu(), os.path.join(save_dir, f"{it:02d}_latent.pt"))

            dec_img = (decoded.clamp(-1,1)*0.5 + 0.5).clamp(0,1)
            pil = tensor_to_pil(dec_img)
            pil.save(os.path.join(save_dir, f"{it:02d}_image.png"))

            # prepare curr_latents and current_image for next iteration via cyclic shift on latent (no re-encode)
            curr_latents = self._cyclic_latent_shift(generated_latent, direction_idx, latent_mask_size)

            print(f"[iter {it}] saved to {save_dir}")

        return 0

# ---------------- main ---------------
if __name__ == "__main__":
    from huggingface_hub import login
    login()
    out_engine = OutpaintEngineFlux(device=DEVICE, lora_path=LORA_PATH)
    # iterate any images in IMG_ROOT
    files = [os.path.join(IMG_ROOT, f) for f in os.listdir(IMG_ROOT) if f.lower().endswith((".png",".jpg",".jpeg"))]
    for p in files:
      for j in range(2):
        img = Image.open(p).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(out_engine.device, dtype=torch.float16)   # [-1,1]
        res = out_engine.generate_iterative(img_t, steps=STEPS, crop_ratio=0.95, iterations=6, direction="up", name=f"{os.path.basename(p)}_{j}")
