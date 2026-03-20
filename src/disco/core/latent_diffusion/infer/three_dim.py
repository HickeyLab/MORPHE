from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from disco.core.latent_diffusion.infer.base import BaseLatentInferencer
from disco.core.latent_diffusion.infer.run_config import ThreeDimImputationRunConfig, ThreeDimImputationWeightSweepConfig
from disco.core.latent_diffusion.train.three_dim import ThreeDimImputationTrainStrategy

from src.disco.core.latent_diffusion.artifact import LatentDiffusionArtifact


class ThreeDimImputationInferencer(BaseLatentInferencer):
    def __init__(
        self, 
        *, 
        artifact: LatentDiffusionArtifact, 
        train_strategy: ThreeDimImputationTrainStrategy, 
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if not isinstance(train_strategy, ThreeDimImputationTrainStrategy):
            raise TypeError(f"train_strategy must be an instance of ThreeDimImputationTrainStrategy, got {type(train_strategy)}")
        
        super().__init__(
            artifact=artifact, 
            train_strategy=train_strategy, 
            pretrained_path=pretrained_path,
            device=device,
            dtype=dtype,
        )

        self.transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    
    def _denormalize(self, x):
        # [-1,1] -> [0,1]
        return (x.clamp(-1, 1) + 1) / 2

    @torch.no_grad()
    def run(self, cfg: ThreeDimImputationRunConfig) -> list[torch.Tensor]:
        # ====== load & preprocess inpaint_images ======
        img_prev = Image.open(cfg.prev_path).convert("RGB")
        img_next = Image.open(cfg.next_path).convert("RGB")

        img_prev = self.transform(img_prev)
        img_next = self.transform(img_next)

        img_prev = img_prev.unsqueeze(0).to(self.device)
        img_next = img_next.unsqueeze(0).to(self.device)

        self.vae.eval()
        self.unet.eval()
        self.cond_proj.eval()

        # --------------------------------------------------
        # 1. Encode prev / next
        # --------------------------------------------------
        latent_prev = self.vae.encode(img_prev).latent_dist.sample() # type: ignore
        latent_next = self.vae.encode(img_next).latent_dist.sample() # type: ignore

        latent_prev = latent_prev * self.scaling_factor
        latent_next = latent_next * self.scaling_factor

        # --------------------------------------------------
        # 2. Build condition (USE NEXT)
        # --------------------------------------------------
        wp = torch.tensor(cfg.w_prev, device=self.device)
        wn = torch.tensor(cfg.w_next, device=self.device)
        condition = self.cond_proj(wp * latent_prev + wn * latent_next)

        # --------------------------------------------------
        # 3. Setup scheduler & timesteps
        # --------------------------------------------------
        self.noise_scheduler.set_timesteps(cfg.num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # --------------------------------------------------
        # 4. Add noise to prev latent (img2img start)
        # --------------------------------------------------
        latents = torch.randn_like(latent_prev)

        # --------------------------------------------------
        # 5. Reverse diffusion
        # --------------------------------------------------
        for t in tqdm(timesteps, leave=False):
            x0_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=condition
            ).sample

            latents = self.noise_scheduler.step(
                model_output=x0_pred,
                timestep=t, # type: ignore
                sample=latents
            ).prev_sample # type: ignore

        # --------------------------------------------------
        # 6. Decode
        # --------------------------------------------------
        latents = latents / self.scaling_factor
        torch.save(latents, str(Path(cfg.out_dir) / cfg.save_latents_name))
        img_mid_pred = self.vae.decode(latents).sample # type: ignore

        x = img_mid_pred.detach().cpu()
        x = self._denormalize(x)
        x = x[0].permute(1, 2, 0)

        if x.max() <= 1.0:
            x_uint8 = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
        else:
            x_uint8 = x.round().clamp(0, 255).to(torch.uint8)

        x_np = x_uint8.numpy()

        Image.fromarray(x_np).save(
            str(Path(cfg.out_dir) / cfg.save_png_name),
            format="PNG"
        )

        return [latents]
        
    @torch.no_grad()
    def run_weight_sweep(self, cfg: ThreeDimImputationWeightSweepConfig):
        w = cfg.start
        while w <= cfg.end + 1e-8:
            w_prev = round(w, 6)
            w_next = 1.0 - w_prev

            lat_name = f"{w_prev:.1f}_{w_next:.1f}.pt"
            png_name = f"mid_{w_prev:.3f}_{w_next:.3f}.png"

            self.run(
                ThreeDimImputationRunConfig(
                    prev_path=cfg.prev_path,
                    next_path=cfg.next_path,
                    out_dir=cfg.out_dir,
                    num_inference_steps=cfg.num_inference_steps,
                    w_prev=w_prev,
                    w_next=w_next,
                    save_latents_name=lat_name,
                    save_png_name=png_name
                )
            )

            print(f"Saved: {lat_name}, {png_name}")
            w += cfg.step