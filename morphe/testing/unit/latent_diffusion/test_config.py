from __future__ import annotations

import pytest

from core.latent_diffusion.train.config import LatentTrainerConfig


# ── defaults ────────────────────────────────────────────────────────


def test_defaults_construct_without_error() -> None:
    cfg = LatentTrainerConfig()

    assert cfg.lr == 2e-5
    assert cfg.batch_size == 8
    assert cfg.epochs == 20
    assert cfg.mixed_precision == "fp16"
    assert cfg.decay_enabled is False


# ── decay validation ────────────────────────────────────────────────


def test_decay_enabled_requires_patience() -> None:
    with pytest.raises(ValueError, match="patience must be provided"):
        LatentTrainerConfig(
            decay_enabled=True,
            lr_decay_every=5,
            lr_decay_factor=0.5,
        )


def test_decay_enabled_requires_lr_decay_every() -> None:
    with pytest.raises(ValueError, match="lr_decay_every must be provided"):
        LatentTrainerConfig(
            decay_enabled=True,
            patience=3,
            lr_decay_factor=0.5,
        )


def test_decay_enabled_requires_lr_decay_factor() -> None:
    with pytest.raises(ValueError, match="lr_decay_factor must be provided"):
        LatentTrainerConfig(
            decay_enabled=True,
            patience=3,
            lr_decay_every=5,
        )


def test_decay_enabled_with_all_params_succeeds() -> None:
    cfg = LatentTrainerConfig(
        decay_enabled=True,
        patience=3,
        lr_decay_every=5,
        lr_decay_factor=0.5,
    )
    assert cfg.decay_enabled is True
    assert cfg.patience == 3


def test_patience_must_be_positive() -> None:
    with pytest.raises(ValueError, match="patience must be > 0"):
        LatentTrainerConfig(patience=0)


def test_lr_decay_every_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lr_decay_every must be > 0"):
        LatentTrainerConfig(lr_decay_every=0)


def test_lr_decay_factor_must_be_between_0_and_1() -> None:
    with pytest.raises(ValueError, match="lr_decay_factor must be between 0 and 1"):
        LatentTrainerConfig(lr_decay_factor=1.5)


def test_lr_decay_factor_zero_rejected() -> None:
    with pytest.raises(ValueError, match="lr_decay_factor must be between 0 and 1"):
        LatentTrainerConfig(lr_decay_factor=0.0)


# ── lr validation ───────────────────────────────────────────────────


def test_lr_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lr must be > 0"):
        LatentTrainerConfig(lr=0.0)


def test_lr_negative_rejected() -> None:
    with pytest.raises(ValueError, match="lr must be > 0"):
        LatentTrainerConfig(lr=-1e-5)


# ── mixed_precision validation ──────────────────────────────────────


def test_mixed_precision_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="mixed_precision must be one of"):
        LatentTrainerConfig(mixed_precision="fp32")


@pytest.mark.parametrize("precision", ["no", "fp16", "bf16"])
def test_mixed_precision_accepts_valid_values(precision: str) -> None:
    cfg = LatentTrainerConfig(mixed_precision=precision)
    assert cfg.mixed_precision == precision


# ── grad_clip validation ────────────────────────────────────────────


def test_grad_clip_negative_rejected() -> None:
    with pytest.raises(ValueError, match="grad_clip must be >= 0"):
        LatentTrainerConfig(grad_clip=-0.1)


def test_grad_clip_zero_accepted() -> None:
    cfg = LatentTrainerConfig(grad_clip=0.0)
    assert cfg.grad_clip == 0.0


# ── batch_size / val_batch_size / epochs ────────────────────────────


@pytest.mark.parametrize("field", ["batch_size", "val_batch_size", "epochs"])
def test_positive_int_fields_reject_zero(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        LatentTrainerConfig(**{field: 0})


@pytest.mark.parametrize("field", ["batch_size", "val_batch_size", "epochs"])
def test_positive_int_fields_reject_negative(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        LatentTrainerConfig(**{field: -1})


# ── encoder kwargs validation ───────────────────────────────────────


@pytest.mark.parametrize(
    "field",
    ["cond_encoder_kwargs", "coord_encoder_kwargs", "bbox_encoder_kwargs"],
)
def test_encoder_kwargs_rejects_non_dict(field: str) -> None:
    with pytest.raises(TypeError, match=f"{field} must be a dict or None"):
        LatentTrainerConfig(**{field: "not_a_dict"})


@pytest.mark.parametrize(
    "field",
    ["cond_encoder_kwargs", "coord_encoder_kwargs", "bbox_encoder_kwargs"],
)
def test_encoder_kwargs_accepts_none(field: str) -> None:
    cfg = LatentTrainerConfig(**{field: None})
    assert getattr(cfg, field) is None


@pytest.mark.parametrize(
    "field",
    ["cond_encoder_kwargs", "coord_encoder_kwargs", "bbox_encoder_kwargs"],
)
def test_encoder_kwargs_accepts_dict(field: str) -> None:
    cfg = LatentTrainerConfig(**{field: {"key": "val"}})
    assert getattr(cfg, field) == {"key": "val"}


# ── pretrained path validation ──────────────────────────────────────


def test_empty_unet_path_rejected() -> None:
    with pytest.raises(TypeError, match="unet_pretrained_path must be a non-empty str"):
        LatentTrainerConfig(unet_pretrained_path="")


def test_empty_ae_path_rejected() -> None:
    with pytest.raises(TypeError, match="ae_pretrained_path must be a non-empty str"):
        LatentTrainerConfig(ae_pretrained_path="")


def test_empty_scheduler_path_rejected() -> None:
    with pytest.raises(TypeError, match="scheduler_pretrained_path must be a non-empty str"):
        LatentTrainerConfig(scheduler_pretrained_path="")
