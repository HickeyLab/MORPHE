from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple
import torch

from torch.utils.data import Dataset

from disco.core.latent_diffusion.diffusion_trainer import DiffusionTrainer


class DiffusionStrategy(ABC):
    requires_coord_encoder: bool = False
    train_num_workers: int = 4
    val_num_workers: int = 2

    # None => "disabled"
    patience: Optional[int] = None
    decay_enabled: Optional[bool] = None
    lr_decay_every: Optional[int] = None
    lr_decay_factor: Optional[float] = None

    @abstractmethod
    def build_dataset(self, root_dir: Path) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError
    
    def train_step(self, trainer: "DiffusionTrainer", batch: Any) -> torch.Tensor:
        pack = self._encode_pack(trainer, batch)

        target_lat = pack["target_lat"]
        B = target_lat.shape[0]

        # 2) noise + timesteps
        scheduler = trainer.noise_scheduler
        noise = torch.randn_like(target_lat)
        t = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (B,),
            device=target_lat.device,
            dtype=torch.long,
        )

        # 3) mask/noise injection
        mask = self._latent_mask(trainer, pack, target_lat)  # None or broadcastable
        noisy_lat = self._noisy_latents(scheduler, target_lat, noise, t, mask)

        # 4) condition tokens
        cond = self._condition_tokens(trainer, pack)

        # 5) predict
        pred = trainer.unet(noisy_lat, t, encoder_hidden_states=cond).sample

        # 6) loss
        return self._loss(pred, pack, noise, mask)

    def validate_step(self, trainer: DiffusionTrainer) -> float:
        raise NotImplementedError
    
    def _encode_latents(self, trainer: "DiffusionTrainer", imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            lat = trainer.vae.encode(imgs).latent_dist.sample()
            return lat * trainer.vae.config.scaling_factor

    def _noisy_latents(self, scheduler, target_lat, noise, t, mask):
        if mask is None:
            return scheduler.add_noise(target_lat, noise, t)
        noisy = scheduler.add_noise(target_lat * mask, noise * mask, t)
        return target_lat * (1 - mask) + noisy
    
    @abstractmethod
    def _encode_pack(self, trainer: "DiffusionTrainer", batch: Any) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _latent_mask(
        self,
        trainer: "DiffusionTrainer",
        pack: dict[str, Any],
        target_lat: torch.Tensor,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    @abstractmethod
    def _condition_tokens(self, trainer: "DiffusionTrainer", pack: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def _loss(
        self,
        pred: torch.Tensor,
        pack: dict[str, Any],
        noise: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError