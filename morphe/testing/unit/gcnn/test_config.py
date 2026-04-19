from __future__ import annotations

import pytest

from core.gcnn.config import GCNNTrainerConfig


# ── defaults ────────────────────────────────────────────────────────


def test_defaults_construct_without_error() -> None:
    cfg = GCNNTrainerConfig()

    assert cfg.label_col == "Cell Type"
    assert cfg.pos_cols == ("x", "y")
    assert cfg.k_neighbors == 20
    assert cfg.batch_size == 1
    assert cfg.hidden_channels == 768
    assert cfg.dropout == 0.1
    assert cfg.alpha == 0.9
    assert cfg.K == 20
    assert cfg.lr == 1e-4
    assert cfg.weight_decay == 1e-5
    assert cfg.epochs == 40


# ── label_col validation ────────────────────────────────────────────


def test_label_col_empty_rejected() -> None:
    with pytest.raises(TypeError, match="label_col must be a non-empty str"):
        GCNNTrainerConfig(label_col="")


# ── pos_cols validation ────────────────────────────────────────────


def test_pos_cols_wrong_length_rejected() -> None:
    with pytest.raises(TypeError, match="pos_cols must be a length-2"):
        GCNNTrainerConfig(pos_cols=("x",))


def test_pos_cols_non_string_rejected() -> None:
    with pytest.raises(TypeError, match="pos_cols must be a length-2"):
        GCNNTrainerConfig(pos_cols=(1, 2))


def test_pos_cols_empty_string_rejected() -> None:
    with pytest.raises(TypeError, match="pos_cols must be a length-2"):
        GCNNTrainerConfig(pos_cols=("x", ""))


def test_pos_cols_accepts_list() -> None:
    cfg = GCNNTrainerConfig(pos_cols=["cx", "cy"])
    assert cfg.pos_cols == ["cx", "cy"]


# ── k_neighbors validation ────────────────────────────────────────


def test_k_neighbors_zero_rejected() -> None:
    with pytest.raises(ValueError, match="k_neighbors must be >= 1"):
        GCNNTrainerConfig(k_neighbors=0)


def test_k_neighbors_float_rejected() -> None:
    with pytest.raises(TypeError, match="k_neighbors must be an int"):
        GCNNTrainerConfig(k_neighbors=5.0)


# ── batch_size validation ─────────────────────────────────────────


def test_batch_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        GCNNTrainerConfig(batch_size=0)


def test_batch_size_negative_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        GCNNTrainerConfig(batch_size=-1)


def test_batch_size_float_rejected() -> None:
    with pytest.raises(TypeError, match="batch_size must be an int"):
        GCNNTrainerConfig(batch_size=2.0)


# ── hidden_channels validation ────────────────────────────────────


def test_hidden_channels_zero_rejected() -> None:
    with pytest.raises(ValueError, match="hidden_channels must be >= 1"):
        GCNNTrainerConfig(hidden_channels=0)


def test_hidden_channels_float_rejected() -> None:
    with pytest.raises(TypeError, match="hidden_channels must be an int"):
        GCNNTrainerConfig(hidden_channels=256.0)


# ── dropout validation ────────────────────────────────────────────


def test_dropout_negative_rejected() -> None:
    with pytest.raises(ValueError, match="dropout must be in"):
        GCNNTrainerConfig(dropout=-0.1)


def test_dropout_one_rejected() -> None:
    with pytest.raises(ValueError, match="dropout must be in"):
        GCNNTrainerConfig(dropout=1.0)


def test_dropout_zero_accepted() -> None:
    cfg = GCNNTrainerConfig(dropout=0.0)
    assert cfg.dropout == 0.0


def test_dropout_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="dropout must be a float"):
        GCNNTrainerConfig(dropout="0.1")


# ── alpha validation ──────────────────────────────────────────────


def test_alpha_zero_rejected() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        GCNNTrainerConfig(alpha=0.0)


def test_alpha_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        GCNNTrainerConfig(alpha=1.1)


def test_alpha_one_accepted() -> None:
    cfg = GCNNTrainerConfig(alpha=1.0)
    assert cfg.alpha == 1.0


def test_alpha_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="alpha must be a float"):
        GCNNTrainerConfig(alpha="0.5")


# ── K validation ──────────────────────────────────────────────────


def test_K_zero_rejected() -> None:
    with pytest.raises(ValueError, match="K must be >= 1"):
        GCNNTrainerConfig(K=0)


def test_K_float_rejected() -> None:
    with pytest.raises(TypeError, match="K must be an int"):
        GCNNTrainerConfig(K=5.0)


# ── lr validation ─────────────────────────────────────────────────


def test_lr_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lr must be > 0"):
        GCNNTrainerConfig(lr=0.0)


def test_lr_negative_rejected() -> None:
    with pytest.raises(ValueError, match="lr must be > 0"):
        GCNNTrainerConfig(lr=-1e-4)


def test_lr_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="lr must be a float"):
        GCNNTrainerConfig(lr="1e-4")


# ── weight_decay validation ───────────────────────────────────────


def test_weight_decay_negative_rejected() -> None:
    with pytest.raises(ValueError, match="weight_decay must be >= 0"):
        GCNNTrainerConfig(weight_decay=-0.01)


def test_weight_decay_zero_accepted() -> None:
    cfg = GCNNTrainerConfig(weight_decay=0.0)
    assert cfg.weight_decay == 0.0


def test_weight_decay_wrong_type_rejected() -> None:
    with pytest.raises(TypeError, match="weight_decay must be a float"):
        GCNNTrainerConfig(weight_decay="0.01")


# ── epochs validation ─────────────────────────────────────────────


def test_epochs_zero_rejected() -> None:
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        GCNNTrainerConfig(epochs=0)


def test_epochs_float_rejected() -> None:
    with pytest.raises(TypeError, match="epochs must be an int"):
        GCNNTrainerConfig(epochs=10.0)
