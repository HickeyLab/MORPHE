from dataclasses import replace
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from disco.core.latent_diffusion.infer.base import BaseLatentInferencer 
from disco.core.latent_diffusion.infer.run_config import GapfillRunConfig
from disco.core.latent_diffusion.train.outpaint import OutpaintTrainStrategy
from src.disco.core.latent_diffusion.artifact import LatentDiffusionArtifact

from src.disco.viz.decoded_img import plot_decoded_image

class GapfillInferencer(BaseLatentInferencer):
    def __init__(
        self, 
        *, 
        artifact: LatentDiffusionArtifact, 
        train_strategy: OutpaintTrainStrategy, 
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        # Require outpaint diffusion strategy
        if not isinstance(train_strategy, OutpaintTrainStrategy):
            raise ValueError("GapfillInferencer requires an OutpaintTrainStrategy")
        
        super().__init__(
            artifact=artifact, 
            train_strategy=train_strategy, 
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )
        
        if self.bbox_encoder is None:
            raise RuntimeError("Gapfill Inferencer requires a bbox encoder in the artifact runtime")
        
        self.transform: torch.Callable[[Image.Image], torch.Tensor] = transforms.Compose([
            transforms.Resize((train_strategy.img_size, train_strategy.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        
    def _create_latent_mask(
        self, 
        bbox: torch.Tensor, 
        latent_shape: Sequence[int]
    ) -> torch.Tensor:
        b, _, lh, lw = latent_shape
        masks = []
        for coords in bbox:
            x1 = coords[0] * lw
            y1 = coords[1] * lh
            x2 = coords[2] * lw
            y2 = coords[3] * lh

            xx, yy = torch.meshgrid(
                torch.arange(lw, device=self.device),
                torch.arange(lh, device=self.device),
            )
            mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
            masks.append(mask)
        return torch.stack(masks).unsqueeze(1)


    def run_one(self, cfg: GapfillRunConfig):        
        with Image.open(cfg.original_dir) as img:
            img = img.convert("RGB")
            image_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Initial stitched image
        current_image = image_tensor.clone()
        b, c, h, w = current_image.shape

        # Define the central gap bbox in normalized coords
        bbox = torch.tensor([[0.0, 0.4375, 1.0, 0.5625]], device=self.device) if cfg.bbox is None else cfg.bbox

        # Initially encode to latent space
        with torch.no_grad():
            current_latent = self.vae.encode(current_image).latent_dist.sample() # type: ignore
            current_latent = current_latent * self.scaling_factor

        for i in range(cfg.iterations):
            print(f"[{i+1}/{cfg.iterations}] Filling central gap in latent space")

            # Step 1: Create latent mask
            latent_mask = self._create_latent_mask(bbox, current_latent.shape)

            # Step 2: Masked latent
            masked_latent = current_latent * (1 - latent_mask)

            # Step 3: Add noise
            noise = torch.randn_like(current_latent)
            noisy_latent = self.noise_scheduler.add_noise(
                masked_latent * latent_mask,
                noise * latent_mask,
                torch.tensor(cfg.steps) # type: ignore
            )
            noisy_latent = masked_latent * (1 - latent_mask) + noisy_latent * latent_mask

            # Step 4: Denoising loop
            self.noise_scheduler.set_timesteps(cfg.steps)
            latent_input = noisy_latent
            
            bbox_encoder = self.bbox_encoder
            if bbox_encoder is None:
                raise RuntimeError("GapfillInferencer requires a bbox encoder in the artifact runtime")
            
            condition = torch.cat([
                self.cond_proj(masked_latent),
                bbox_encoder.unsqueeze(1).expand(-1, 64, -1)
            ], dim=-1)

            cnt = 0
            for t in self.noise_scheduler.timesteps:
                cnt += 1
                latent_input = latent_input * latent_mask + masked_latent * (1 - latent_mask)
                with torch.no_grad():
                    noise_pred = self.unet(latent_input, t, encoder_hidden_states=condition).sample
                latent_input = self.noise_scheduler.step(noise_pred, t, latent_input).prev_sample # type: ignore
                if cnt == cfg.steps - 1:
                    generated_latent = latent_input * latent_mask + masked_latent * (1 - latent_mask)  ## SAVE THIS AS JSON FILE AND PUT INTO DECODE INFERENCE
                    generated_img = self.vae.decode(generated_latent / self.scaling_factor).sample  # type: ignore
                    preview = ((generated_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

            if i == cfg.iterations - 1:
                # Final decode
                with torch.no_grad():
                    generated_latent = latent_input * latent_mask + masked_latent * (1 - latent_mask)
                    generated_img = self.vae.decode(generated_latent / self.scaling_factor).sample  # type: ignore
                    
                    preview = ((generated_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                    
                    save_dir = Path(cfg.save_dir)
                    run_dir = save_dir / cfg.save_name
                    run_dir.mkdir(parents=True, exist_ok=True)

                    latent_path = run_dir / f"{i}_latent.pt"
                    image_path = run_dir / f"{i}_image.png"

                    torch.save(generated_latent.detach().cpu(), latent_path)
                    Image.fromarray(preview.astype(np.uint8)).save(image_path)
                    
                    if cfg.show_plot:
                        plot_decoded_image(
                            preview=preview,
                            iteration=i,
                            figsize=cfg.plot_fig_size,
                            title=cfg.plot_title
                        )
                        
                    return generated_latent

            # Step 5: merge latent
            # extract the newly generated center region from latent_input
            new_patch_latent = latent_input[..., :, :, 28:36]   # latent pixels [28:36] = 64 image pixels

            # split into left and right halves (each 4 latent pixels)
            left_latent_patch = new_patch_latent[..., :, :, :4]   # (1, C, H, 4)
            right_latent_patch = new_patch_latent[..., :, :, 4:]  # (1, C, H, 4)

            # crop sides from current latent
            cropped_latent = current_latent[..., :, :, 4:-4]      # remove 4 latent pixels from each side → shape (1, C, H, 56)

            # split cropped latent into left and right parts
            cropped_left = cropped_latent[..., :, :, :24]         # left part
            cropped_right = cropped_latent[..., :, :, -24:]       # right part

            # create white gap latent (8 latent pixels = 64 image pixels)
            white_gap_latent = torch.zeros(
                (1, current_latent.shape[1], current_latent.shape[2], 8),
                device=self.device
            )

            # concatenate parts:
            # cropped_left | left_patch | white_gap | right_patch | cropped_right
            stitched_latent = torch.cat([
                cropped_left,
                left_latent_patch,
                white_gap_latent,
                right_latent_patch,
                cropped_right
            ], dim=-1)

            # Crop back to 64 latent pixels width
            start = (stitched_latent.shape[-1] - 64) // 2
            stitched_latent = stitched_latent[..., :, :, start:start + 64]

            # Update for next iteration
            current_latent = stitched_latent

            # decode and show intermediate result
            with torch.no_grad():
                decoded_img = self.vae.decode(current_latent / self.scaling_factor).sample

            preview = (decoded_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            """Image.fromarray((preview * 255).astype(np.uint8)).save(f"iteration_{i+1}.png")"""
            
            if cfg.show_plot:
                plot_decoded_image(
                    preview=preview,
                    iteration=i,
                    figsize=cfg.plot_fig_size,
                    title=cfg.plot_title
                )

        # In case iterations == 0 or loop ends unexpectedly
        return current_latent
        
    def run(self, cfg: GapfillRunConfig) -> list[torch.Tensor]:
        cfg.validate()
        
        results: list[torch.Tensor] = []

        files = sorted(p for p in cfg.original_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png") # type: ignore

        for fpath in files:
            per_file_save_name = str(Path(cfg.save_name) / fpath.stem)
            file_cfg = replace(
                cfg,
                original_dir=fpath,
                save_name=per_file_save_name,
            )
            res = self.run_one(file_cfg)
            results.append(res)

        return results