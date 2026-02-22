from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple, Optional

import torch
from diffusers import DDPMScheduler

from disco.core.pixel_diffusion.models import LatentAdapter, UNet512


@dataclass(frozen=True)
class PixelDiffusionRuntime:
    adapter: torch.nn.Module
    unet512: torch.nn.Module
    noise_scheduler: DDPMScheduler
    device: torch.device
    dtype: torch.dtype
    

@dataclass(frozen=True)
class PixelDiffusionTrainerArtifact:
    adapter_state_dict: Mapping[str, torch.Tensor]
    unet512_state_dict: Mapping[str, torch.Tensor]
    adapter_kwargs: Mapping[str, Any]
    unet_kwargs: Mapping[str, Any]
    train_index: str
    val_index: str
    bs: int
    lr: float
    ae_pretrained: str
    enable_epoch_visualiations: bool
    optimizer_betas: Tuple[float, float]
    optimizer_weight_decay: float
    optimizer_state_dict: Optional[Mapping[str, Any]] = None
    epoch: int = 0
    global_step: int = 0

    def build_models(
        self,
        *,
        device: torch.device | str | None = None,
        eval_mode: bool = True,
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        
        adapter = LatentAdapter(**dict(self.adapter_kwargs))
        unet512 = UNet512(**dict(self.unet_kwargs))

        adapter.load_state_dict(self.adapter_state_dict, strict=True)
        unet512.load_state_dict(self.unet512_state_dict, strict=True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        adapter = adapter.to(device)
        unet512 = unet512.to(device)

        if eval_mode:
            adapter.eval()
            unet512.eval()

        return adapter, unet512

    def build_inference_runtime(
        self,
        *,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        num_inference_steps: int = 150,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PixelDiffusionRuntime:
        # device
        if device is None:
            device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_ = torch.device(device)

        # dtype
        if dtype is None:
            if device_.type == "cuda":
                dtype_ = torch.float16
            else:
                dtype_ = torch.float32
        else:
            dtype_ = dtype

        # build + load weights
        adapter = LatentAdapter(**dict(self.adapter_kwargs))
        unet512 = UNet512(**dict(self.unet_kwargs))

        adapter.load_state_dict(self.adapter_state_dict, strict=True)
        unet512.load_state_dict(self.unet512_state_dict, strict=True)

        adapter = adapter.to(device=device_, dtype=dtype_)
        unet512 = unet512.to(device=device_, dtype=dtype_)

        adapter.eval()
        unet512.eval()

        # scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
        noise_scheduler.set_timesteps(num_inference_steps, device=device_)
        noise_scheduler.config.prediction_type = "sample"

        return PixelDiffusionRuntime(
            adapter=adapter,
            unet512=unet512,
            noise_scheduler=noise_scheduler,
            device=device_,
            dtype=dtype_,
        )

    def save(self, path: str | Path) -> None:
        """
        Serialize the artifact as a single torch file (like GCNNArtifact.save).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # ensure tensors are CPU for portability
        adapter_cpu = {k: v.detach().cpu() for k, v in self.adapter_state_dict.items()}
        unet_cpu = {k: v.detach().cpu() for k, v in self.unet512_state_dict.items()}

        payload = {
            "adapter_state_dict": adapter_cpu,
            "unet512_state_dict": unet_cpu,
            "adapter_kwargs": dict(self.adapter_kwargs),
            "unet_kwargs": dict(self.unet_kwargs),
            "train_index": str(self.train_index),
            "val_index": str(self.val_index),
            "bs": int(self.bs),
            "lr": float(self.lr),
            "ae_pretrained": str(self.ae_pretrained),
            "enable_epoch_visualiations": bool(self.enable_epoch_visualiations),
            "optimizer_betas": tuple(float(x) for x in self.optimizer_betas),
            "optimizer_weight_decay": float(self.optimizer_weight_decay),
            "optimizer_state_dict": dict(self.optimizer_state_dict) if self.optimizer_state_dict is not None else None,
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
        }

        torch.save(payload, str(path))

    @staticmethod
    def load(path: str | Path) -> "PixelDiffusionTrainerArtifact":
        path = Path(path)

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")

        required = (
            "adapter_state_dict",
            "unet512_state_dict",
            "adapter_kwargs",
            "unet_kwargs",
            "train_index",
            "val_index",
            "bs",
            "lr",
            "ae_pretrained",
            "enable_epoch_visualiations",
            "optimizer_betas",
            "optimizer_weight_decay",
            "epoch",
            "global_step",
        )
        for k in required:
            if k not in payload:
                raise ValueError(f"Invalid PixelDiffusionTrainerArtifact file: missing key '{k}'")

        optimizer_state = payload.get("optimizer_state_dict", None)

        return PixelDiffusionTrainerArtifact(
            adapter_state_dict=payload["adapter_state_dict"],
            unet512_state_dict=payload["unet512_state_dict"],
            adapter_kwargs=payload["adapter_kwargs"],
            unet_kwargs=payload["unet_kwargs"],
            train_index=str(payload["train_index"]),
            val_index=str(payload["val_index"]),
            bs=int(payload["bs"]),
            lr=float(payload["lr"]),
            ae_pretrained=str(payload["ae_pretrained"]),
            enable_epoch_visualiations=bool(payload["enable_epoch_visualiations"]),
            optimizer_betas=tuple(float(x) for x in payload["optimizer_betas"]),
            optimizer_weight_decay=float(payload["optimizer_weight_decay"]),
            optimizer_state_dict=optimizer_state,
            epoch=int(payload["epoch"]),
            global_step=int(payload["global_step"]),
        )