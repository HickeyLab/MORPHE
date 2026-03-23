from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Self

import torch
from diffusers import DDPMScheduler  # type: ignore

from disco.core.pixel_diffusion.inferencer import PixelDiffusionInferencer
from disco.core.pixel_diffusion.models import LatentAdapter, UNet512
from disco.utils import resolve_device, resolve_dtype, set_config_attr


@dataclass(frozen=True)
class PixelDiffusionArtifact:
    """
    Serializable artifact for the pixel diffusion stage.

    This artifact stores model weights, model-construction kwargs, and the
    minimal training metadata needed to rebuild inference components or resume
    training state when applicable.
    """

    adapter_state_dict: Mapping[str, torch.Tensor]
    unet_state_dict: Mapping[str, torch.Tensor]
    adapter_kwargs: Mapping[str, Any]
    unet_kwargs: Mapping[str, Any]
    scheduler_pretrained: str
    scheduler_num_inference_steps: int

    train_index: str
    val_index: str

    bs: int
    lr: float
    
    optimizer_betas: tuple[float, float]
    optimizer_weight_decay: float
    optimizer_state_dict: Mapping[str, Any] | None = None

    best_epoch: int = 0
    best_global_step: int = 0

    def _build_loaded_models(self, device: torch.device, dtype: torch.dtype) -> tuple[LatentAdapter, UNet512]:
        """
        Instantiate models from stored kwargs and load artifact weights.

        Returns:
            A fully loaded adapter and UNet pair on CPU.
        """
        adapter = LatentAdapter(**dict(self.adapter_kwargs))
        unet512 = UNet512(**dict(self.unet_kwargs))

        adapter.load_state_dict(self.adapter_state_dict, strict=True)
        unet512.load_state_dict(self.unet_state_dict, strict=True)
        
        adapter = adapter.to(device=device, dtype=dtype)
        unet512 = unet512.to(device=device, dtype=dtype)

        adapter.eval()
        unet512.eval()

        return adapter, unet512

    def build_inference_components(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        scheduler_num_inference_steps: int | None = None,
    ) -> tuple[LatentAdapter, UNet512, DDPMScheduler, torch.device, torch.dtype]:
        """
        Build inference-ready models and scheduler.

        Args:
            pretrained_path: Diffusers checkpoint path used to construct the
                scheduler.
            num_inference_steps: Number of scheduler timesteps for inference.
            device: Target device for inference components.
            dtype: Target dtype for inference models. When omitted, defaults to
                float16 on CUDA and float32 otherwise.

        Returns:
            A tuple of:
                - loaded adapter
                - loaded UNet
                - configured scheduler
                - resolved device
                - resolved dtype
        """
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(device=resolved_device, dtype=dtype)

        adapter, unet512 = self._build_loaded_models(device=resolved_device, dtype=resolved_dtype)

        noise_scheduler = DDPMScheduler.from_pretrained(
            self.scheduler_pretrained,
            subfolder="scheduler",
        )
        noise_scheduler.set_timesteps(
            self.scheduler_num_inference_steps if not scheduler_num_inference_steps else scheduler_num_inference_steps, 
            device=resolved_device
        )

        # Keep scheduler behavior aligned with the expected pixel diffusion output.
        set_config_attr(noise_scheduler.config, "prediction_type", "sample")

        return adapter, unet512, noise_scheduler, resolved_device, resolved_dtype

    def save(self, path: str | Path) -> None:
        """
        Serialize the artifact to a single torch file.

        State dict tensors are moved to CPU before saving to improve portability
        across devices and environments.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        adapter_cpu = {
            key: value.detach().cpu()
            for key, value in self.adapter_state_dict.items()
        }
        unet_cpu = {
            key: value.detach().cpu()
            for key, value in self.unet_state_dict.items()
        }

        payload = {
            "adapter_state_dict": adapter_cpu,
            "unet_state_dict": unet_cpu,
            "adapter_kwargs": dict(self.adapter_kwargs),
            "unet_kwargs": dict(self.unet_kwargs),
            "scheduler_pretrained": self.scheduler_pretrained,
            "scheduler_num_inference_steps": self.scheduler_num_inference_steps,
            "train_index": self.train_index,
            "val_index": self.val_index,
            "bs": self.bs,
            "lr": self.lr,
            "optimizer_betas": tuple(float(x) for x in self.optimizer_betas),
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_state_dict": (
                dict(self.optimizer_state_dict)
                if self.optimizer_state_dict is not None
                else None
            ),
            "best_epoch": self.best_epoch,
            "best_global_step": self.best_global_step,
        }

        torch.save(payload, str(path))

    @staticmethod
    def _coerce_optimizer_betas(value: Any) -> tuple[float, float]:
        """Validate and normalize optimizer betas from a loaded payload."""
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("optimizer_betas must be a sequence of length 2")
        return float(value[0]), float(value[1])

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load a serialized artifact from disk.

        Args:
            path: Path to a previously saved artifact file.

        Returns:
            A reconstructed PixelDiffusionArtifact instance.

        Raises:
            ValueError: If the file payload is missing required keys.
        """
        path = Path(path)

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")

        required_keys = (
            "adapter_state_dict",
            "unet_state_dict",
            "adapter_kwargs",
            "unet_kwargs",
            "train_index",
            "val_index",
            "bs",
            "lr",
            "scheduler_pretrained",
            "scheduler_num_inference_steps",
            "optimizer_betas",
            "optimizer_weight_decay",
            "best_epoch",
            "best_global_step",
        )
        for key in required_keys:
            if key not in payload:
                raise ValueError(
                    f"Invalid PixelDiffusionArtifact file: missing key '{key}'"
                )

        optimizer_state = payload.get("optimizer_state_dict")

        return cls(
            adapter_state_dict=payload["adapter_state_dict"],
            unet_state_dict=payload["unet_state_dict"],
            adapter_kwargs=payload["adapter_kwargs"],
            unet_kwargs=payload["unet_kwargs"],
            scheduler_pretrained=payload["scheduler_pretrained"],
            scheduler_num_inference_steps=payload["scheduler_num_inference_steps"],
            train_index=str(payload["train_index"]),
            val_index=str(payload["val_index"]),
            bs=int(payload["bs"]),
            lr=float(payload["lr"]),
            optimizer_betas=cls._coerce_optimizer_betas(
                payload["optimizer_betas"]
            ),
            optimizer_weight_decay=float(payload["optimizer_weight_decay"]),
            optimizer_state_dict=optimizer_state,
            best_epoch=int(payload["best_epoch"]),
            best_global_step=int(payload["best_global_step"]),
        )
        
    def build_inferencer(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        num_inference_steps: int | None = None,
    ) -> PixelDiffusionInferencer:
        """
        Build a PixelDiffusionInferencer from this artifact.

        Args:
            pretrained_path: Diffusers checkpoint path used to construct the
                scheduler.
            num_inference_steps: Number of scheduler timesteps for inference.
            device: Target device for inference components. If ``None``, a default
                 device is resolved automatically.
            dtype: Target dtype for inference models. If ``None``, defaults to
                float16 on CUDA and float32 otherwise.
        """
        adapter, unet512, noise_scheduler, resolved_device, resolved_dtype = (
            self.build_inference_components(
                scheduler_num_inference_steps=num_inference_steps if num_inference_steps is not None else self.scheduler_num_inference_steps,
                device=device,
                dtype=dtype,
            )
        )

        return PixelDiffusionInferencer(
            artifact=self,
            adapter=adapter,
            unet512=unet512,
            noise_scheduler=noise_scheduler,
            device=resolved_device,
            dtype=resolved_dtype,
        )