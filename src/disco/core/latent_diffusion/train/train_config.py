from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LatentTrainerConfig:
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    pretrained: str = "runwayml/stable-diffusion-v1-5"
    lr: float = 2e-5
    mixed_precision: str = "fp16"
    grad_clip: float = 1.0
    cond_encoder_kwargs: dict | None = None
    coord_encoder_kwargs: dict | None = None
    bbox_encoder_kwargs: dict | None = None
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 20

    def validate(self) -> None:
        if not isinstance(self.save_dir, str) or not self.save_dir:
            raise TypeError("save_dir must be a non-empty str.")
        if not isinstance(self.save_best_only, bool):
            raise TypeError("save_best_only must be a bool.")

        if not isinstance(self.pretrained, str) or not self.pretrained:
            raise TypeError("pretrained must be a non-empty str.")

        if not isinstance(self.lr, (int, float)):
            raise TypeError("lr must be a float > 0.")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be > 0.")

        if not isinstance(self.mixed_precision, str) or not self.mixed_precision:
            raise TypeError("mixed_precision must be a non-empty str.")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of {'no', 'fp16', 'bf16'}.")

        if not isinstance(self.grad_clip, (int, float)):
            raise TypeError("grad_clip must be a float >= 0 (or None if you support that).")
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

        for name, val in (
            ("cond_encoder_kwargs", self.cond_encoder_kwargs),
            ("coord_encoder_kwargs", self.coord_encoder_kwargs),
            ("bbox_encoder_kwargs", self.bbox_encoder_kwargs),
        ):
            if val is not None and not isinstance(val, dict):
                raise TypeError(f"{name} must be a dict or None.")