from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LatentTrainerConfig:
    """
    Configuration for training the latent diffusion model.

    This config controls:
    - optimization and training loop behavior
    - checkpointing and output management
    - pretrained backbone initialization
    - optional encoder construction for conditioning

    Learning-rate decay:
        decay_enabled: Whether to enable learning-rate decay.
        patience: Number of epochs with no improvement before triggering decay.
        lr_decay_every: Frequency (in epochs) at which to apply LR decay.
        lr_decay_factor: Multiplicative factor applied to the learning rate.

    Checkpointing / output:
        save_dir: Directory where checkpoints and artifacts are saved.
        save_best_only: Whether to keep only the best checkpoint based on validation loss.

    Pretrained backbone:
        unet_pretrained_path: Path or HuggingFace identifier for the UNet model.
        ae_pretrained_path: Path or HuggingFace identifier for the VAE (autoencoder).
        scheduler_pretrained_path: Path or identifier for the diffusion scheduler.

    Optimization / training loop:
        lr: Learning rate for optimizer.
        mixed_precision: Precision mode ("fp16", "bf16", or "no").
        grad_clip: Maximum gradient norm for clipping.
        batch_size: Training batch size.
        val_batch_size: Validation batch size.
        epochs: Number of training epochs.

    Conditioning encoders (optional):
        cond_encoder_kwargs: Keyword arguments for building the conditioning encoder.
        coord_encoder_kwargs: Keyword arguments for coordinate encoder (used in spatial tasks).
        bbox_encoder_kwargs: Keyword arguments for bounding-box encoder (used in gapfill/outpaint).

    Notes:
        - The actual encoder types (cond, coord, bbox) are determined by the
          selected training task (e.g., inpaint, gapfill, 3D imputation).
        - Pretrained paths are used to initialize diffusion components via
          `from_pretrained(...)` when available.
        - Mixed precision is handled via Accelerate during training.

        Learning-rate decay behavior:
        - If `decay_enabled=True`, all decay parameters (`patience`,
          `lr_decay_every`, `lr_decay_factor`) must be provided.
        - If `decay_enabled=False`, decay parameters are ignored.

        Task-specific behavior:
        - Not all inference modes use all encoders:
            - Inpaint: cond encoder only
            - Gapfill / Outpaint: cond + bbox (+ possibly coord)
            - 3D imputation: specialized conditioning (e.g., cond3d)

    Typical usage:
        cfg = LatentTrainerConfig(
            lr=2e-5,
            batch_size=8,
            epochs=20,
            mixed_precision="fp16",
        )

    Performance tips:
        - If training is unstable, reduce `lr` or increase `grad_clip`.
        - If running out of memory, reduce `batch_size` or disable mixed precision fallback.
        - If underfitting, increase `epochs` or adjust conditioning encoder capacity.
    """
    # Learning-rate decay
    decay_enabled: bool = False
    patience: int | None = None
    lr_decay_every: int | None = None
    lr_decay_factor: float | None = None

    # Checkpointing / output
    save_dir: str | Path = "checkpoints"
    save_best_only: bool = True

    # Pretrained backbone
    unet_pretrained_path: str = "runwayml/stable-diffusion-v1-5"
    ae_pretrained_path: str = "runwayml/stable-diffusion-v1-5"
    scheduler_pretrained_path: str = "runwayml/stable-diffusion-v1-5"

    # Optimization / training loop
    lr: float = 2e-5
    mixed_precision: str = "fp16"
    grad_clip: float = 1.0
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 20

    # Optional encoder construction kwargs
    cond_encoder_kwargs: dict[str, Any] | None = None
    coord_encoder_kwargs: dict[str, Any] | None = None
    bbox_encoder_kwargs: dict[str, Any] | None = None
    
    def __post_init__(self) -> None:
        """Validate config immediately after construction."""
        self.validate()
        object.__setattr__(self, "save_dir", Path(self.save_dir))

    def validate(self) -> None:
        """Validate that all config values are internally consistent."""
        # ------------------------------------------------------------------
        # Decay configuration
        # ------------------------------------------------------------------
        if not isinstance(self.decay_enabled, bool):
            raise TypeError("decay_enabled must be a bool.")

        if self.decay_enabled:
            if self.patience is None:
                raise ValueError("patience must be provided when decay_enabled is True.")
            if self.lr_decay_every is None:
                raise ValueError("lr_decay_every must be provided when decay_enabled is True.")
            if self.lr_decay_factor is None:
                raise ValueError("lr_decay_factor must be provided when decay_enabled is True.")

        if self.patience is not None:
            if not isinstance(self.patience, int):
                raise TypeError("patience must be an int or None.")
            if self.patience <= 0:
                raise ValueError("patience must be > 0.")

        if self.lr_decay_every is not None:
            if not isinstance(self.lr_decay_every, int):
                raise TypeError("lr_decay_every must be an int or None.")
            if self.lr_decay_every <= 0:
                raise ValueError("lr_decay_every must be > 0.")

        if self.lr_decay_factor is not None:
            if not isinstance(self.lr_decay_factor, (int, float)):
                raise TypeError("lr_decay_factor must be a float or None.")
            if not (0.0 < float(self.lr_decay_factor) < 1.0):
                raise ValueError("lr_decay_factor must be between 0 and 1.")

        # ------------------------------------------------------------------
        # Checkpointing / output
        # ------------------------------------------------------------------
        if not isinstance(self.save_dir, str) or not self.save_dir:
            raise TypeError("save_dir must be a non-empty str.")
        if not isinstance(self.save_best_only, bool):
            raise TypeError("save_best_only must be a bool.")

        # ------------------------------------------------------------------
        # Pretrained backbone
        # ------------------------------------------------------------------
        if not isinstance(self.unet_pretrained_path, str) or not self.unet_pretrained_path:
            raise TypeError("unet_pretrained_path must be a non-empty str.")
        if not isinstance(self.ae_pretrained_path, str) or not self.ae_pretrained_path:
            raise TypeError("ae_pretrained_path must be a non-empty str.")
        if not isinstance(self.scheduler_pretrained_path, str) or not self.scheduler_pretrained_path:
            raise TypeError("scheduler_pretrained_path must be a non-empty str.")

        # ------------------------------------------------------------------
        # Optimization / training loop
        # ------------------------------------------------------------------
        if not isinstance(self.lr, (int, float)):
            raise TypeError("lr must be a float > 0.")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be > 0.")

        if not isinstance(self.mixed_precision, str) or not self.mixed_precision:
            raise TypeError("mixed_precision must be a non-empty str.")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of {'no', 'fp16', 'bf16'}.")

        if not isinstance(self.grad_clip, (int, float)):
            raise TypeError("grad_clip must be a float >= 0.")
        if float(self.grad_clip) < 0.0:
            raise ValueError("grad_clip must be >= 0.")

        for name, val in (
            ("batch_size", self.batch_size),
            ("val_batch_size", self.val_batch_size),
            ("epochs", self.epochs),
        ):
            if not isinstance(val, int):
                raise TypeError(f"{name} must be an int.")
            if val <= 0:
                raise ValueError(f"{name} must be > 0.")

        # ------------------------------------------------------------------
        # Encoder kwargs
        # ------------------------------------------------------------------
        for name, val in (
            ("cond_encoder_kwargs", self.cond_encoder_kwargs),
            ("coord_encoder_kwargs", self.coord_encoder_kwargs),
            ("bbox_encoder_kwargs", self.bbox_encoder_kwargs),
        ):
            if val is not None and not isinstance(val, dict):
                raise TypeError(f"{name} must be a dict or None.")