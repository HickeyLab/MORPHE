from __future__ import annotations

import pytest

from core.pixel_diffusion.config import PixelDiffusionTrainerConfig


# ── defaults ────────────────────────────────────────────────────────


def test_defaults_construct_without_error(tmp_path: object) -> None:
    cfg = PixelDiffusionTrainerConfig()

    assert cfg.lr == 1e-5
    assert cfg.train_batch_size == 4
    assert cfg.epochs == 30
    assert cfg.mixed_precision == "fp16"
    assert cfg.patience == 5


# ── lr validation ───────────────────────────────────────────────────


def test_lr_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lr must be a positive number"):
        PixelDiffusionTrainerConfig(lr=0.0)


def test_lr_negative_rejected() -> None:
    with pytest.raises(ValueError, match="lr must be a positive number"):
        PixelDiffusionTrainerConfig(lr=-1e-5)


# ── optimizer_betas validation ──────────────────────────────────────


def test_optimizer_betas_wrong_type() -> None:
    with pytest.raises(TypeError, match="optimizer_betas must be a tuple"):
        PixelDiffusionTrainerConfig(optimizer_betas=[0.9, 0.999])


def test_optimizer_betas_wrong_length() -> None:
    with pytest.raises(TypeError, match="optimizer_betas must be a tuple"):
        PixelDiffusionTrainerConfig(optimizer_betas=(0.9,))


# ── optimizer_weight_decay validation ───────────────────────────────


def test_weight_decay_negative_rejected() -> None:
    with pytest.raises(ValueError, match="optimizer_weight_decay must be a number >= 0"):
        PixelDiffusionTrainerConfig(optimizer_weight_decay=-0.01)


def test_weight_decay_zero_accepted() -> None:
    cfg = PixelDiffusionTrainerConfig(optimizer_weight_decay=0.0)
    assert cfg.optimizer_weight_decay == 0.0


# ── mixed_precision validation ──────────────────────────────────────


def test_mixed_precision_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="mixed_precision must be one of"):
        PixelDiffusionTrainerConfig(mixed_precision="fp32")


@pytest.mark.parametrize("precision", ["no", "fp16", "bf16"])
def test_mixed_precision_accepts_valid(precision: str) -> None:
    cfg = PixelDiffusionTrainerConfig(mixed_precision=precision)
    assert cfg.mixed_precision == precision


# ── data loading validation ─────────────────────────────────────────


def test_train_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="train_batch_size must be a positive int"):
        PixelDiffusionTrainerConfig(train_batch_size=0)


def test_train_batch_size_negative_rejected() -> None:
    with pytest.raises(ValueError, match="train_batch_size"):
        PixelDiffusionTrainerConfig(train_batch_size=-1)


def test_num_workers_negative_rejected() -> None:
    with pytest.raises(ValueError, match="train_num_workers must be a non-negative int"):
        PixelDiffusionTrainerConfig(train_num_workers=-1)


def test_num_workers_zero_accepted() -> None:
    cfg = PixelDiffusionTrainerConfig(train_num_workers=0)
    assert cfg.train_num_workers == 0


# ── pretrained paths ────────────────────────────────────────────────


def test_empty_ae_pretrained_rejected() -> None:
    with pytest.raises(ValueError, match="ae_pretrained must be a non-empty str"):
        PixelDiffusionTrainerConfig(ae_pretrained="")


def test_empty_scheduler_pretrained_rejected() -> None:
    with pytest.raises(ValueError, match="scheduler_pretrained must be a non-empty str"):
        PixelDiffusionTrainerConfig(scheduler_pretrained="")


# ── kwargs validation ───────────────────────────────────────────────


@pytest.mark.parametrize("field", ["adapter_kwargs", "unet_kwargs"])
def test_kwargs_rejects_non_dict(field: str) -> None:
    with pytest.raises(TypeError, match=f"{field} must be dict or None"):
        PixelDiffusionTrainerConfig(**{field: "bad"})


@pytest.mark.parametrize("field", ["adapter_kwargs", "unet_kwargs"])
def test_kwargs_accepts_none(field: str) -> None:
    cfg = PixelDiffusionTrainerConfig(**{field: None})
    assert getattr(cfg, field) is None


# ── training loop validation ────────────────────────────────────────


@pytest.mark.parametrize("field", ["epochs", "patience"])
def test_positive_int_fields_reject_zero(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive int"):
        PixelDiffusionTrainerConfig(**{field: 0})


@pytest.mark.parametrize("field", ["epochs", "patience"])
def test_positive_int_fields_reject_negative(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive int"):
        PixelDiffusionTrainerConfig(**{field: -1})
