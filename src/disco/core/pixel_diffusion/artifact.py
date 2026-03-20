from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Self

import torch
from diffusers import DDPMScheduler  # type: ignore

from disco.core.pixel_diffusion.models import LatentAdapter, UNet512
from disco.utils import set_config_attr


@dataclass(frozen=True)
class PixelDiffusionArtifact:
    """
    Serializable artifact for the pixel diffusion stage.

    This artifact stores model weights, model-construction kwargs, and the
    minimal training metadata needed to rebuild inference components or resume
    training state when applicable.
    """

    adapter_state_dict: Mapping[str, torch.Tensor]
    unet512_state_dict: Mapping[str, torch.Tensor]
    adapter_kwargs: Mapping[str, Any]
    unet_kwargs: Mapping[str, Any]

    train_index: str
    val_index: str

    bs: int
    lr: float
    ae_pretrained: str
    enable_epoch_visualizations: bool
    optimizer_betas: tuple[float, float]
    optimizer_weight_decay: float
    optimizer_state_dict: Mapping[str, Any] | None = None

    epoch: int = 0
    global_step: int = 0

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        """Resolve a user-provided device or choose a sensible default."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_dtype(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> torch.dtype:
        """Resolve an inference dtype based on device when none is provided."""
        if dtype is not None:
            return dtype
        return torch.float16 if device.type == "cuda" else torch.float32

    def _build_loaded_models(self) -> tuple[LatentAdapter, UNet512]:
        """
        Instantiate models from stored kwargs and load artifact weights.

        Returns:
            A fully loaded adapter and UNet pair on CPU.
        """
        adapter = LatentAdapter(**dict(self.adapter_kwargs))
        unet512 = UNet512(**dict(self.unet_kwargs))

        adapter.load_state_dict(self.adapter_state_dict, strict=True)
        unet512.load_state_dict(self.unet512_state_dict, strict=True)

        return adapter, unet512

    def build_models(
        self,
        *,
        device: torch.device | str | None = None,
        eval_mode: bool = True,
    ) -> tuple[LatentAdapter, UNet512]:
        """
        Rebuild the trained models from the artifact.

        Args:
            device: Target device for the models. Defaults to CUDA when
                available, otherwise CPU.
            eval_mode: Whether to put the models into evaluation mode.

        Returns:
            The rebuilt adapter and UNet models on the requested device.
        """
        resolved_device = self._resolve_device(device)
        adapter, unet512 = self._build_loaded_models()

        adapter = adapter.to(resolved_device)
        unet512 = unet512.to(resolved_device)

        if eval_mode:
            adapter.eval()
            unet512.eval()

        return adapter, unet512

    def build_inference_components(
        self,
        *,
        pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        num_inference_steps: int = 150,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
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
        resolved_device = self._resolve_device(device)
        resolved_dtype = self._resolve_dtype(device=resolved_device, dtype=dtype)

        adapter, unet512 = self._build_loaded_models()
        adapter = adapter.to(device=resolved_device, dtype=resolved_dtype)
        unet512 = unet512.to(device=resolved_device, dtype=resolved_dtype)

        adapter.eval()
        unet512.eval()

        noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_path,
            subfolder="scheduler",
        )
        noise_scheduler.set_timesteps(num_inference_steps, device=resolved_device)

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
            for key, value in self.unet512_state_dict.items()
        }

        payload = {
            "adapter_state_dict": adapter_cpu,
            "unet512_state_dict": unet_cpu,
            "adapter_kwargs": dict(self.adapter_kwargs),
            "unet_kwargs": dict(self.unet_kwargs),
            "train_index": self.train_index,
            "val_index": self.val_index,
            "bs": self.bs,
            "lr": self.lr,
            "ae_pretrained": self.ae_pretrained,
            "enable_epoch_visualizations": self.enable_epoch_visualizations,
            "optimizer_betas": tuple(float(x) for x in self.optimizer_betas),
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_state_dict": (
                dict(self.optimizer_state_dict)
                if self.optimizer_state_dict is not None
                else None
            ),
            "epoch": self.epoch,
            "global_step": self.global_step,
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
            "unet512_state_dict",
            "adapter_kwargs",
            "unet_kwargs",
            "train_index",
            "val_index",
            "bs",
            "lr",
            "ae_pretrained",
            "enable_epoch_visualizations",
            "optimizer_betas",
            "optimizer_weight_decay",
            "epoch",
            "global_step",
        )
        for key in required_keys:
            if key not in payload:
                raise ValueError(
                    f"Invalid PixelDiffusionArtifact file: missing key '{key}'"
                )

        optimizer_state = payload.get("optimizer_state_dict")

        return cls(
            adapter_state_dict=payload["adapter_state_dict"],
            unet512_state_dict=payload["unet512_state_dict"],
            adapter_kwargs=payload["adapter_kwargs"],
            unet_kwargs=payload["unet_kwargs"],
            train_index=str(payload["train_index"]),
            val_index=str(payload["val_index"]),
            bs=int(payload["bs"]),
            lr=float(payload["lr"]),
            ae_pretrained=str(payload["ae_pretrained"]),
            enable_epoch_visualizations=bool(
                payload["enable_epoch_visualizations"]
            ),
            optimizer_betas=cls._coerce_optimizer_betas(
                payload["optimizer_betas"]
            ),
            optimizer_weight_decay=float(payload["optimizer_weight_decay"]),
            optimizer_state_dict=optimizer_state,
            epoch=int(payload["epoch"]),
            global_step=int(payload["global_step"]),
        )