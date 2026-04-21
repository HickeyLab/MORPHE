from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PixelDiffusionTrainerConfig:
    """
    Configuration for training the pixel diffusion (cascade) model.

    This stage learns to decode latent representations into full-resolution
    RGB outputs using a UNet-based diffusion model conditioned on latents.

    This config controls:
    - optimizer and training loop behavior
    - dataloader performance
    - pretrained diffusion components
    - visualization and evaluation outputs

    Core optimization:
        lr: Learning rate for the optimizer.
        optimizer_betas: Beta coefficients for Adam optimizer.
        optimizer_weight_decay: L2 weight decay applied during optimization.

    Runtime:
        mixed_precision: Precision mode used during training ("fp16", "bf16", or "no").

    Data loading:
        train_num_workers: Number of workers for training DataLoader.
        train_batch_size: Batch size for training.
        val_num_workers: Number of workers for validation DataLoader.
        val_batch_size: Batch size for validation.

    Model / pretrained components:
        ae_pretrained: Path or HuggingFace identifier for the VAE used to encode latents.
        scheduler_pretrained: Path or identifier for the diffusion scheduler.
        scheduler_num_inference_steps: Number of denoising steps used during sampling.
        adapter_kwargs: Optional kwargs for constructing the latent adapter module.
        unet_kwargs: Optional kwargs for constructing the UNet model.

    Output / logging:
        vis_dir: Directory where training visualizations are saved.
        comp_eval_save_dir: Directory where composition evaluation outputs are saved.

    Training loop:
        epochs: Number of training epochs.
        patience: Early stopping patience based on validation performance.

    Notes:
        - Training operates on precomputed latent datasets (not raw images).
        - The model learns to map latent inputs → RGB outputs via diffusion.
        - The adapter transforms latent space into a form compatible with the UNet.
        - The scheduler controls the diffusion noise schedule and sampling process.
        - Visualization outputs typically include intermediate reconstructions and
          qualitative comparisons between original and predicted images.

    Expected inputs:
        - Precomputed dataset containing:
            - latent tensors (e.g., shape [4, 64, 64])
            - target images (e.g., shape [3, 512, 512])

    Performance tips:
        - If training is slow, reduce `scheduler_num_inference_steps` or batch size.
        - If running out of memory, lower `train_batch_size` or disable mixed precision fallback.
        - If outputs are blurry, increase epochs or adjust UNet capacity via `unet_kwargs`.
        - If conditioning is weak, tune `adapter_kwargs`.

    Typical usage:
        cfg = PixelDiffusionTrainerConfig(
            lr=1e-5,
            train_batch_size=4,
            epochs=30,
            scheduler_num_inference_steps=200,
        )
    """
    # -------------------------------
    # Core optimization
    # -------------------------------
    lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_weight_decay: float = 1e-5

    # Runtime
    mixed_precision: str = "fp16"

    # Data loading
    train_num_workers: int = 4
    train_batch_size: int = 4
    val_num_workers: int = 2
    val_batch_size: int = 2

    # Model / pretrained sources
    ae_pretrained: str = "runwayml/stable-diffusion-v1-5"
    scheduler_pretrained: str = "runwayml/stable-diffusion-v1-5"
    scheduler_num_inference_steps: int = 200
    adapter_kwargs: dict[str, int] | None = None
    unet_kwargs: dict[str, int] | None = None

    # Output
    vis_dir: str | Path = "visualizations"
    comp_eval_save_dir: str | Path = "comp_eval"

    # Training loop
    epochs: int = 30
    patience: int = 5
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "vis_dir", Path(self.vis_dir))
        object.__setattr__(self, "comp_eval_save_dir", Path(self.comp_eval_save_dir))
        self.validate()

    def validate(self) -> None:
        # -------------------------------
        # Core optimization
        # -------------------------------
        if not isinstance(self.train_batch_size, int) or self.train_batch_size <= 0:
            raise ValueError(f"train_batch_size must be a positive int, got {self.train_batch_size!r}")

        if not isinstance(self.lr, (int, float)) or float(self.lr) <= 0:
            raise ValueError(f"lr must be a positive number, got {self.lr!r}")

        if (
            not isinstance(self.optimizer_betas, tuple)
            or len(self.optimizer_betas) != 2
            or not all(isinstance(x, (int, float)) for x in self.optimizer_betas)
        ):
            raise TypeError("optimizer_betas must be a tuple[float, float]")

        if not isinstance(self.optimizer_weight_decay, (int, float)) or float(self.optimizer_weight_decay) < 0:
            raise ValueError("optimizer_weight_decay must be a number >= 0")

        # -------------------------------
        # Runtime
        # -------------------------------
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of {'no', 'fp16', 'bf16'}")

        # -------------------------------
        # Data loading
        # -------------------------------
        for name, val in (
            ("train_num_workers", self.train_num_workers),
            ("val_num_workers", self.val_num_workers),
            ("val_batch_size", self.val_batch_size),
            ("train_batch_size", self.train_batch_size),
        ):
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative int, got {val!r}")

        # -------------------------------
        # Model / pretrained sources
        # -------------------------------
        if not isinstance(self.ae_pretrained, str) or not self.ae_pretrained:
            raise ValueError("ae_pretrained must be a non-empty str")

        if not isinstance(self.scheduler_pretrained, str) or not self.scheduler_pretrained:
            raise ValueError("scheduler_pretrained must be a non-empty str")

        for name, val in (("adapter_kwargs", self.adapter_kwargs), ("unet_kwargs", self.unet_kwargs)):
            if val is not None and not isinstance(val, dict):
                raise TypeError(f"{name} must be dict or None, got {type(val).__name__}")

        # -------------------------------
        # Output
        # -------------------------------
        if not (isinstance(self.vis_dir, Path) and str(self.vis_dir)):
            raise ValueError("vis_dir must be a non-empty Path")

        if not (isinstance(self.comp_eval_save_dir, Path) and str(self.comp_eval_save_dir)):
            raise ValueError("comp_eval_save_dir must be a non-empty Path")

        # -------------------------------
        # Training loop
        # -------------------------------
        for name, val in (
            ("epochs", self.epochs),
            ("patience", self.patience),
        ):
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive int, got {val!r}")