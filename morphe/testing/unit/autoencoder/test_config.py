from __future__ import annotations

from pathlib import Path

import pytest

from core.autoencoder.config import AutoencoderTrainerConfig


# ── defaults ────────────────────────────────────────────────────────


def test_defaults_construct_without_error() -> None:
    cfg = AutoencoderTrainerConfig()

    assert cfg.save_best_only is True
    assert cfg.val_ratio == 0.1
    assert cfg.batch_size == 4096
    assert cfg.num_workers == 4
    assert cfg.input_cols is None
    assert cfg.bottleneck_dim == 3
    assert cfg.hidden_dim == 512
    assert cfg.num_epochs == 100
    assert cfg.lr == 1e-6
    assert cfg.alpha == 0.1


def test_save_dir_coerced_to_path() -> None:
    cfg = AutoencoderTrainerConfig(save_dir="my_dir")
    assert isinstance(cfg.save_dir, Path)
    assert cfg.save_dir == Path("my_dir")


# ── val_ratio validation ───────────────────────────────────────────


def test_val_ratio_zero_rejected() -> None:
    with pytest.raises(ValueError, match="val_ratio must be in"):
        AutoencoderTrainerConfig(val_ratio=0.0)


def test_val_ratio_one_rejected() -> None:
    with pytest.raises(ValueError, match="val_ratio must be in"):
        AutoencoderTrainerConfig(val_ratio=1.0)


def test_val_ratio_negative_rejected() -> None:
    with pytest.raises(ValueError, match="val_ratio must be in"):
        AutoencoderTrainerConfig(val_ratio=-0.1)


def test_val_ratio_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="val_ratio must be a float"):
        AutoencoderTrainerConfig(val_ratio="0.1")


# ── int field validation ──────────────────────────────────────────


@pytest.mark.parametrize("field", ["batch_size", "num_workers", "bottleneck_dim", "hidden_dim", "num_epochs"])
def test_int_fields_reject_float(field: str) -> None:
    with pytest.raises(TypeError, match=f"{field} must be an int"):
        AutoencoderTrainerConfig(**{field: 1.5})


def test_batch_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        AutoencoderTrainerConfig(batch_size=0)


def test_batch_size_negative_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        AutoencoderTrainerConfig(batch_size=-1)


def test_num_workers_negative_rejected() -> None:
    with pytest.raises(ValueError, match="num_workers must be >= 0"):
        AutoencoderTrainerConfig(num_workers=-1)


def test_num_workers_zero_accepted() -> None:
    cfg = AutoencoderTrainerConfig(num_workers=0)
    assert cfg.num_workers == 0


def test_bottleneck_dim_zero_rejected() -> None:
    with pytest.raises(ValueError, match="bottleneck_dim must be > 0"):
        AutoencoderTrainerConfig(bottleneck_dim=0)


def test_hidden_dim_zero_rejected() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be > 0"):
        AutoencoderTrainerConfig(hidden_dim=0)


def test_num_epochs_zero_rejected() -> None:
    with pytest.raises(ValueError, match="num_epochs must be > 0"):
        AutoencoderTrainerConfig(num_epochs=0)


# ── lr validation ──────────────────────────────────────────────────


def test_lr_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lr must be > 0"):
        AutoencoderTrainerConfig(lr=0.0)


def test_lr_negative_rejected() -> None:
    with pytest.raises(ValueError, match="lr must be > 0"):
        AutoencoderTrainerConfig(lr=-1e-5)


def test_lr_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="lr must be a float"):
        AutoencoderTrainerConfig(lr="1e-4")


# ── alpha validation ──────────────────────────────────────────────


def test_alpha_negative_rejected() -> None:
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        AutoencoderTrainerConfig(alpha=-0.1)


def test_alpha_zero_accepted() -> None:
    cfg = AutoencoderTrainerConfig(alpha=0.0)
    assert cfg.alpha == 0.0


def test_alpha_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="alpha must be a float"):
        AutoencoderTrainerConfig(alpha="0.1")


# ── input_cols validation ─────────────────────────────────────────


def test_input_cols_accepts_none() -> None:
    cfg = AutoencoderTrainerConfig(input_cols=None)
    assert cfg.input_cols is None


def test_input_cols_accepts_valid_list() -> None:
    cfg = AutoencoderTrainerConfig(input_cols=["prob_a", "prob_b"])
    assert list(cfg.input_cols) == ["prob_a", "prob_b"]


def test_input_cols_empty_rejected() -> None:
    with pytest.raises(ValueError, match="input_cols cannot be empty"):
        AutoencoderTrainerConfig(input_cols=[])


def test_input_cols_non_string_entries_rejected() -> None:
    with pytest.raises(TypeError, match="All input_cols entries must be strings"):
        AutoencoderTrainerConfig(input_cols=["a", 123])


def test_input_cols_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="input_cols must be a sequence"):
        AutoencoderTrainerConfig(input_cols="not_a_list")
