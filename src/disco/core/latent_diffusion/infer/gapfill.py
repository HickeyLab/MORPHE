import os
from pathlib import Path
from typing import ClassVar, Sequence
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from disco.core.latent_diffusion.artifact import LatentDiffuserArtifact
from disco.core.latent_diffusion.strategy.outpaint import OutpaintDiffusion

from disco.viz.decoded_img import plot_decoded_image

from .base import LatentDiffusionInferencer, InferenceResult


class GapfillInferencer(LatentDiffusionInferencer):
    REQUIRED_STRATEGY_NAME: ClassVar[str] = "outpaint"

    def __init__(
        self, 
        *, 
        artifact: LatentDiffuserArtifact, 
        strategy: OutpaintDiffusion, 
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        # Require outpaint diffusion strategy
        if getattr(strategy, "strategy_name", None) != self.REQUIRED_STRATEGY_NAME:
            raise TypeError(
                f"GapfillInferencer requires strategy_name='{self.REQUIRED_STRATEGY_NAME}', "
                f"got {getattr(strategy, 'strategy_name', None)!r}"
            )
        super().__init__(
            artifact=artifact, 
            strategy=strategy, 
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((strategy.img_size, strategy.img_size)),
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
    
    def _validate_run_args(
        self,
        original_dir: str | Path,
        save_dir: str | Path,
        *,
        steps: int,
        iterations: int,
        plot_fig_size: tuple[int, int] | None,
    ) -> tuple[Path, Path]:
        if original_dir is None or save_dir is None:
            raise RuntimeError(
                "GapfillInferencer requires original_dir and save_dir but one of them is None."
            )

        original_dir = Path(original_dir)
        save_dir = Path(save_dir)

        if not original_dir.exists():
            raise FileNotFoundError(f"original_dir does not exist: {original_dir}")
        
        save_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("`steps` must be a positive integer.")

        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("`iterations` must be a positive integer.")

        if plot_fig_size is not None:
            if (
                not isinstance(plot_fig_size, tuple)
                or len(plot_fig_size) != 2
                or not all(isinstance(x, int) and x > 0 for x in plot_fig_size)
            ):
                raise ValueError(
                    "`plot_fig_size` must be tuple[int, int] with positive values."
                )

        if self.vae is None or self.unet is None:
            raise RuntimeError(
                "Inferencer requires loaded VAE and UNet."
            )

        return original_dir, save_dir
    
    def _validate_weight_sweep_args(
        self,
        prev_path: str | Path,
        next_path: str | Path,
        out_dir: str | Path,
        *,
        num_inference_steps: int,
        start: float,
        end: float,
        step: float,
    ) -> tuple[Path, Path, Path]:
        if prev_path is None or next_path is None or out_dir is None:
            raise RuntimeError("ThreeDimensionalInferer requires prev_path, next_path, and out_dir but one is None.")

        prev_path = Path(prev_path)
        next_path = Path(next_path)
        out_dir = Path(out_dir)

        if not prev_path.exists() or not prev_path.is_file():
            raise FileNotFoundError(f"prev_path does not exist or is not a file: {prev_path}")
        if not next_path.exists() or not next_path.is_file():
            raise FileNotFoundError(f"next_path does not exist or is not a file: {next_path}")

        out_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("`num_inference_steps` must be a positive integer.")

        if not isinstance(start, (int, float)) or not 0.0 <= float(start) <= 1.0:
            raise ValueError("`start` must be in [0, 1].")

        if not isinstance(end, (int, float)) or not 0.0 <= float(end) <= 1.0:
            raise ValueError("`end` must be in [0, 1].")

        if float(start) >= float(end):
            raise ValueError("`start` must be less than `end`.")

        if not isinstance(step, (int, float)) or float(step) <= 0:
            raise ValueError("`step` must be a positive float.")

        if float(step) > (float(end) - float(start)):
            raise ValueError("`step` is larger than the sweep range.")

        if self.vae is None or self.unet is None:
            raise RuntimeError("Inferencer requires loaded VAE and UNet.")
        if self.cond_proj is None:
            raise RuntimeError("Inferencer requires loaded cond_proj.")
        if self.noise_scheduler is None:
            raise RuntimeError("Inferencer requires loaded noise_scheduler.")

        return prev_path, next_path, out_dir


    def run_one(
        self,
        original_file_path: str | Path,
        save_dir: str | Path,
        save_name: str = "default",
        steps: int = 200,
        iterations: int = 10,
        show_plot: bool = False,
        plot_title: str | None = None,
        plot_fig_size: tuple[int, int] | None = None,
        bbox: torch.Tensor | None = None,
    ):
        original_file_path, save_dir = self._validate_run_args(
            original_file_path,
            save_dir,
            steps=steps,
            iterations=iterations,
            plot_fig_size=plot_fig_size
        )
        
        with Image.open(original_file_path) as img:
            img = img.convert("RGB")
            image_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Initial stitched image
        current_image = image_tensor.clone()
        b, c, h, w = current_image.shape

        # Define the central gap bbox in normalized coords
        bbox = torch.tensor([[0.0, 0.4375, 1.0, 0.5625]], device=self.device) if bbox is None else bbox

        # Initially encode to latent space
        with torch.no_grad():
            current_latent = self.vae.encode(current_image).latent_dist.sample()
            current_latent = current_latent * self.scaling_factor

        for i in range(iterations):
            print(f"[{i+1}/{iterations}] Filling central gap in latent space")

            # Step 1: Create latent mask
            latent_mask = self._create_latent_mask(bbox, current_latent.shape)

            # Step 2: Masked latent
            masked_latent = current_latent * (1 - latent_mask)

            # Step 3: Add noise
            noise = torch.randn_like(current_latent)
            noisy_latent = self.noise_scheduler.add_noise(
                masked_latent * latent_mask,
                noise * latent_mask,
                torch.tensor(steps)
            )
            noisy_latent = masked_latent * (1 - latent_mask) + noisy_latent * latent_mask

            # Step 4: Denoising loop
            self.noise_scheduler.set_timesteps(steps)
            latent_input = noisy_latent

            condition = torch.cat([
                self.cond_proj(masked_latent),
                self.coord_encoder(bbox).unsqueeze(1).expand(-1, 64, -1)
            ], dim=-1)

            cnt = 0
            for t in self.noise_scheduler.timesteps:
                cnt += 1
                latent_input = latent_input * latent_mask + masked_latent * (1 - latent_mask)
                with torch.no_grad():
                    noise_pred = self.unet(latent_input, t, encoder_hidden_states=condition).sample
                latent_input = self.noise_scheduler.step(noise_pred, t, latent_input).prev_sample
                if cnt == steps - 1:
                    generated_latent = latent_input * latent_mask + masked_latent * (1 - latent_mask)  ## SAVE THIS AS JSON FILE AND PUT INTO DECODE INFERENCE
                    generated_img = self.vae.decode(generated_latent / self.scaling_factor).sample
                    preview = ((generated_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

            if i == iterations - 1:
                # Final decode
                with torch.no_grad():
                    generated_latent = latent_input * latent_mask + masked_latent * (1 - latent_mask)
                    generated_img = self.vae.decode(generated_latent / self.scaling_factor).sample
                    
                    preview = ((generated_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                    
                    save_dir = Path(save_dir)
                    run_dir = save_dir / save_name
                    run_dir.mkdir(parents=True, exist_ok=True)

                    latent_path = run_dir / f"{i}_latent.pt"
                    image_path = run_dir / f"{i}_image.png"

                    torch.save(generated_latent.detach().cpu(), latent_path)
                    Image.fromarray(preview.astype(np.uint8)).save(image_path)
                    
                    if show_plot:
                        plot_decoded_image(
                            preview=preview,
                            iteration=i,
                            figsize=plot_fig_size,
                            title=plot_title
                        )
                        
                    return InferenceResult(
                        image=generated_img,
                        latents=generated_latent,
                        extras={
                            "image_save_path": save_dir,
                            "preview": preview,
                            "iteration": i,
                        },
                    )

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
            
            if show_plot:
                plot_decoded_image(
                    preview=preview,
                    iteration=i,
                    figsize=plot_fig_size,
                    title=plot_title
                )

        # In case iterations == 0 or loop ends unexpectedly
        return InferenceResult(
            image=decoded_img if "decoded_img" in locals() else current_image,
            latents=current_latent,
        )
        
    def run(
        self,
        original_dir: str | Path,
        save_dir: str | Path,
        *,
        save_name: str = "default",
        steps: int = 200,
        iterations: int = 10,
        show_plot: bool = False,
        plot_title: str | None = None,
        plot_fig_size: tuple[int, int] | None = None,
        bbox: torch.Tensor | None = None,
    ) -> list["InferenceResult"]:
        original_dir, save_dir = self._validate_run_args(
            original_dir,
            save_dir,
            steps=steps,
            iterations=iterations,
            plot_fig_size=plot_fig_size
        )

        results: list[InferenceResult] = []

        files = sorted(p for p in original_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")

        for fpath in files:
            per_file_save_name = str(Path(save_name) / fpath.stem)

            res = self.run_one(
                original_file_path=fpath,
                save_dir=save_dir,
                save_name=per_file_save_name,
                steps=steps,
                iterations=iterations,
                show_plot=show_plot,
                plot_title=plot_title,
                plot_fig_size=plot_fig_size,
                bbox=bbox,
            )

            if res.extras is not None:
                res.extras.setdefault("original_file_path", str(fpath))
            results.append(res)

        return results
    
    @torch.no_grad()
    def run_weight_sweep(
        self,
        *,
        prev_path: str | Path,
        next_path: str | Path,
        out_dir: str | Path = "./outputs",
        num_inference_steps: int = 200,
        start: float = 0.1,
        end: float = 0.9,
        step: float = 0.1,
    ):
        prev_path, next_path, out_dir = self._validate_weight_sweep_args(
            prev_path=prev_path,
            next_path=next_path,
            out_dir=out_dir,
            num_inference_steps=num_inference_steps,
            start=start,
            end=end,
            step=step
        )

        w = start
        while w <= end + 1e-8:
            w_prev = round(w, 6)
            w_next = 1.0 - w_prev

            lat_name = f"{w_prev:.1f}_{w_next:.1f}.pt"
            png_name = f"mid_{w_prev:.3f}_{w_next:.3f}.png"

            self.run_one(
                prev_path=prev_path,
                next_path=next_path,
                out_dir=out_dir,
                num_inference_steps=num_inference_steps,
                w_prev=w_prev,
                w_next=w_next,
                save_latents_name=lat_name,
                save_png_name=png_name,
            )

            print(f"Saved: {lat_name}, {png_name}")

            w += step
