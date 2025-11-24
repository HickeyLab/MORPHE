import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DConditionModel
from accelerate import Accelerator
from torchvision import transforms
from PIL import Image
from OutpaintTrainer_newoutpaint import OutpaintTrainer
from diffusers import AutoencoderKL

class OutpaintEngine:
    def __init__(self, device="cuda"):
        """
        Initialize the outpainting engine.

        Args:
            device: Computation device (default: "cuda").
        """
        self.device = device

        # Load model components
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet"
        )

        self.accelerator = Accelerator(
            mixed_precision='fp16'
        )

        # Initialize scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )

        # Direction encoder
        self.directions = ['right', 'left', 'down', 'up']
        self.coord_encoder = nn.Sequential(nn.Linear(4, 32), nn.GELU())

        # Condition projection layer
        self.cond_proj = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(256, 736, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.15)
        )

        # Set eval mode
        self.vae.eval()
        self.vae.to(device)
        self.unet.eval()
        self.unet.to(device)
        self.cond_proj.eval()
        self.cond_proj.to(device)
        self.coord_encoder.eval()
        self.coord_encoder.to(device)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def _extract_old_region(self, image_tensor, direction, crop_ratio):
        """Apply zero mask to input image
        Args:
            image_tensor: Input tensor (1,3,H,W)
            direction: Mask direction (0-3)
            crop_ratio: Ratio of preserved area
        Returns:
            masked_tensor: Zero-masked image tensor
            mask_params: Parameters for latent mask calculation
        """
        b, c, h, w = image_tensor.shape
        masked = image_tensor.clone()

        if direction in [0, 1]:  # Horizontal directions
            crop_w = int(w * crop_ratio)
            if direction == 0:  # Preserve left
                masked = masked[..., :, :, :crop_w]
            else:  # Preserve right
                masked = masked[..., :, :, w - crop_w:]
        else:  # Vertical directions
            crop_h = int(h * crop_ratio)
            if direction == 2:  # Preserve top
                masked = masked[..., :crop_h, :]
            else:  # Preserve bottom
                masked = masked[..., h - crop_h:, :]

        return masked

    def _create_latent_mask_from_bbox(self, bbox, latent_shape):
        b, _, lh, lw = latent_shape
        masks = []
        for coords in bbox:
            x1 = int(coords[0] * lw)
            y1 = int(coords[1] * lh)
            x2 = int(coords[2] * lw)
            y2 = int(coords[3] * lh)
            mask = torch.zeros((1, lh, lw), device=self.device)
            mask[:, x1:x2, y1:y2] = 1
            masks.append(mask)
        return torch.stack(masks)

    def _create_latent_mask(self, bbox, latent_shape):
        b, _, lh, lw = latent_shape
        masks = []
        for coords in bbox:
            x1 = coords[0] * lw
            y1 = coords[1] * lh
            x2 = coords[2] * lw
            y2 = coords[3] * lh

            xx, yy = torch.meshgrid(
                torch.arange(lw, device=self.accelerator.device),
                torch.arange(lh, device=self.accelerator.device))
            mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()

            masks.append(mask)
        return torch.stack(masks).unsqueeze(1)

    def compute_mean_std(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)  # [1, 4, 1, 1]
        std = x.std(dim=(2, 3), keepdim=True)  # [1, 4, 1, 1]
        return mean, std

    def generate_iterative(self, image_tensor, steps=200, crop_ratio=0.97, iterations=10, direction="right"):
        """
        Iteratively expand image along a direction using bbox + coord_encoder logic.
        """
        direction_idx = self.directions.index(direction)
        current_image = image_tensor.clone()
        b, c, h, w = current_image.shape
        lh, lw = int(64), int(64)
        #initial_latent = self.vae.encode(current_image).latent_dist.sample()
        #initial_latent = initial_latent * self.vae.config.scaling_factor
        initial_mean, initial_std = self.compute_mean_std(current_image)

        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        crop_lw, crop_lh = int(lw * crop_ratio), int(lh * crop_ratio)
        mask_size = w - crop_w if direction in ["left", "right"] else h - crop_h
        latent_mask_size = lw - crop_lw if direction in ["left", "right"] else lh - crop_lh

        # Extract preserved region to initialize stitched image
        preserved_region = self._extract_old_region(current_image, direction_idx, crop_ratio)
        stitched = preserved_region
        current_latent = None
        # Step 1: Create bbox (normalized)
        if direction == "right":
            bbox = torch.tensor([[0.0, crop_ratio, 1.0, 1.0]], device=self.device)
        elif direction == "left":
            bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0 - crop_ratio]], device=self.device)
        elif direction == "down":
            bbox = torch.tensor([[crop_ratio, 0.0, 1.0, 1.0]], device=self.device)
        else:
            bbox = torch.tensor([[0.0, 0.0, 1.0 - crop_ratio, 1.0]], device=self.device)

        for i in range(iterations):
            print(f"[{i + 1}/{iterations}] Expanding → {direction.upper()}")
            if i == 0:
                # Step 2: Create masked image
                mask = torch.zeros_like(current_image)
                x1 = int(bbox[0][0] * w)
                y1 = int(bbox[0][1] * h)
                x2 = int(bbox[0][2] * w)
                y2 = int(bbox[0][3] * h)
                mask[:, :, x1:x2, y1:y2] = 1
                masked_img = current_image * (1 - mask)
                print(mask)

                # Step 3: Encode condition
                with torch.no_grad():
                    masked_latents = self.vae.encode(masked_img).latent_dist.sample()
                    masked_latents = masked_latents * self.vae.config.scaling_factor
            else:
                masked_latents = current_latent

            # Step 4: Add noise and create latent_mask
            latent_mask = self._create_latent_mask(bbox, masked_latents.shape)
            print(latent_mask)

            noise = torch.randn_like(masked_latents)
            noisy_latents = self.noise_scheduler.add_noise(masked_latents * latent_mask, noise * latent_mask,
                                                           torch.tensor(steps))
            noisy_latents = masked_latents * (1 - latent_mask) + noisy_latents * latent_mask

            # Step 5: Denoising loop
            self.noise_scheduler.set_timesteps(steps)
            latent_input = noisy_latents

            condition = torch.cat([
                self.cond_proj(masked_latents).flatten(2).transpose(1, 2),
                self.coord_encoder(bbox).unsqueeze(1).expand(-1, 64, -1)
            ], dim=-1)

            for t in self.noise_scheduler.timesteps:
                latent_input = latent_input * latent_mask + masked_latents * (1 - latent_mask)
                with torch.no_grad():
                    noise_pred = self.unet(latent_input, t, encoder_hidden_states=condition).sample
                latent_input = self.noise_scheduler.step(noise_pred, t, latent_input).prev_sample

            # Step 6: Decode image
            with torch.no_grad():
                generated_latent = masked_latents * (1 - latent_mask) + latent_input * latent_mask
                test = masked_latents * (1 - latent_mask)
                generated_img = self.vae.decode(generated_latent / self.vae.config.scaling_factor).sample
                test = self.vae.decode(test / self.vae.config.scaling_factor).sample

            plt.figure(figsize=(8, 8))
            plt.imshow((test[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1))
            plt.title(f"Test {i + 1} Result")
            plt.axis("off")
            plt.show()

            #tgt_mean, tgt_std = self.compute_mean_std(generated_img)
            #generated_img = (generated_img - tgt_mean) / tgt_std * initial_std + initial_mean
            # Step 7: Extract new patch and update
            plt.figure(figsize=(8, 8))
            plt.imshow((generated_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1))
            plt.title(f"Generated {i + 1} Result")
            plt.axis("off")
            plt.show()

            new_patch = self._extract_new_region(generated_img, direction_idx, mask_size)
            stitched = self._stitch_image(stitched, new_patch, direction_idx)
            current_image = self._cyclic_shift(generated_img, direction_idx, mask_size)
            current_latent = self._cyclic_latent_shift(generated_latent, direction_idx, latent_mask_size)

            # Display intermediate
            plt.figure(figsize=(8, 8))
            plt.imshow((stitched[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1))
            plt.title(f"Iteration {i + 1} Result")
            plt.axis("off")
            plt.show()

        return stitched[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5

    def _extract_new_region(self, generated, direction, mask_size):
        """extract new region"""
        if direction == 0:  # right
            return generated[..., :, :, -mask_size:]
        elif direction == 1:  # left
            return generated[..., :, :, :mask_size]
        elif direction == 2:  # down
            return generated[..., :, -mask_size:, :]
        elif direction == 3:  # up
            return generated[..., :, :mask_size, :]

    def _extract_new_latent_region(self, generated, direction, mask_size):
        """extract new region"""

        if direction == 0:  # right
            return generated[..., :, :, -mask_size:]
        elif direction == 1:  # left
            return generated[..., :, :, :mask_size]
        elif direction == 2:  # down
            return generated[..., :, -mask_size:, :]
        elif direction == 3:  # up
            return generated[..., :, :mask_size, :]

    def _stitch_image(self, combined, generated_patch, direction):
        """
        Append newly generated regions to the stitched image to create a final large output.

        Args:
            combined: The accumulated image tensor containing all past expansions.
            generated_patch: The new generated patch to append.
            direction: The expansion direction.

        Returns:
            Updated stitched image tensor.
        """
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
        """do iteration（"for example generating right region"：ABCDE → BCDEA）"""
        if direction == 0:  # right
            return torch.cat([generated[..., :, :, mask_size:],
                              generated[..., :, :, :mask_size]], dim=-1)
        elif direction == 1:  # left
            return torch.cat([generated[..., :, :, -mask_size:],
                              generated[..., :, :, :-mask_size]], dim=-1)
        elif direction == 2:  # down
            return torch.cat([generated[..., :, mask_size:, :],
                              generated[..., :, :mask_size, :]], dim=-2)
        elif direction == 3:  # up
            return torch.cat([generated[..., :, -mask_size:, :],
                              generated[..., :, :-mask_size, :]], dim=-2)

    def _cyclic_latent_shift(self, generated, direction, mask_size):
        """do iteration（"for example generating right region"：ABCDE → BCDEA）"""
        if direction == 0:  # right
            return torch.cat([generated[..., :, :, mask_size:],
                              generated[..., :, :, :mask_size]], dim=-1)
        elif direction == 1:  # left
            return torch.cat([generated[..., :, :, -mask_size:],
                              generated[..., :, :, :-mask_size]], dim=-1)
        elif direction == 2:  # down
            return torch.cat([generated[..., :, mask_size:, :],
                              generated[..., :, :mask_size, :]], dim=-2)
        elif direction == 3:  # up
            return torch.cat([generated[..., :, -mask_size:, :],
                              generated[..., :, :-mask_size, :]], dim=-2)

if __name__ == "__main__":
    outpaint_engine = OutpaintEngine()
    trainer = OutpaintTrainer()
    trainer.accelerator.load_state("D:/checkpoint-newprompt")
    outpaint_engine.unet = trainer.unet
    outpaint_engine.cond_proj = trainer.cond_proj
    outpaint_engine.coord_encoder = trainer.coord_encoder
    # Preprocess input image
    img = Image.open("C:/Users/18983/Desktop/traindata/region_B009_Right_embeddings.png").convert("RGB")
    #B008_Duodenum_plot.png B009_Right_plot.png B011_Proximal jejunum_plot.png B012_Mid jejunum_plot.png
    img = outpaint_engine.transform(img).unsqueeze(0).to(outpaint_engine.accelerator.device)
    # Run iterative expansion, keeping a stitched version
    stitched_image = outpaint_engine.generate_iterative(img,steps=100, iterations=10, direction="down")
    plt.figure(figsize=(20, 8))
    plt.imshow(stitched_image)
    plt.title("Final Stitched Image (Large Canvas)")
    plt.axis("off")
    plt.show()
