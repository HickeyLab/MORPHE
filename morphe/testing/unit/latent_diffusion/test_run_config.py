from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.latent_diffusion.inference.run_config import (
    InpaintRunConfig,
    GapfillRunConfig,
    OutpaintRunConfig,
    ThreeDimImputationRunConfig,
    ThreeDimImputationWeightSweepConfig,
)


# ── InpaintRunConfig ────────────────────────────────────────────────


def test_inpaint_validate_raises_for_missing_input_dir(tmp_path: Path) -> None:
    cfg = InpaintRunConfig(
        input_dir=tmp_path / "nonexistent",
        mask_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="input_dir does not exist"):
        cfg.validate()


def test_inpaint_validate_raises_for_missing_mask_dir(tmp_path: Path) -> None:
    cfg = InpaintRunConfig(
        input_dir=tmp_path,
        mask_dir=tmp_path / "nonexistent",
    )
    with pytest.raises(FileNotFoundError, match="mask_dir does not exist"):
        cfg.validate()


def test_inpaint_validate_raises_for_invalid_num_steps(tmp_path: Path) -> None:
    cfg = InpaintRunConfig(
        input_dir=tmp_path,
        mask_dir=tmp_path,
        num_steps=0,
    )
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        cfg.validate()


def test_inpaint_validate_raises_for_bad_fig_size(tmp_path: Path) -> None:
    cfg = InpaintRunConfig(
        input_dir=tmp_path,
        mask_dir=tmp_path,
        plot_fig_size=(0, 5),
    )
    with pytest.raises(ValueError, match="plot_fig_size values must be > 0"):
        cfg.validate()


def test_inpaint_validate_succeeds_for_valid_config(tmp_path: Path) -> None:
    cfg = InpaintRunConfig(
        input_dir=tmp_path,
        mask_dir=tmp_path,
        num_steps=100,
    )
    cfg.validate()  # should not raise


# ── GapfillRunConfig ────────────────────────────────────────────────


def test_gapfill_validate_raises_for_missing_input_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="input_dir does not exist"):
        GapfillRunConfig(
            input_dir=tmp_path / "nonexistent",
            save_dir=tmp_path / "out",
        )


def test_gapfill_validate_raises_for_empty_save_name(tmp_path: Path) -> None:
    cfg = GapfillRunConfig.__new__(GapfillRunConfig)
    cfg.input_dir = tmp_path
    cfg.save_dir = tmp_path / "out"
    cfg.save_name = "   "
    cfg.num_steps = 100
    cfg.iterations = 5
    cfg.show_plot = False
    cfg.plot_title = "test"
    cfg.plot_fig_size = (6, 6)
    cfg.bbox = None

    with pytest.raises(ValueError, match="save_name must be non-empty"):
        cfg.validate()


def test_gapfill_validate_raises_for_invalid_iterations(tmp_path: Path) -> None:
    cfg = GapfillRunConfig.__new__(GapfillRunConfig)
    cfg.input_dir = tmp_path
    cfg.save_dir = tmp_path / "out"
    cfg.save_name = "test"
    cfg.num_steps = 100
    cfg.iterations = 0
    cfg.show_plot = False
    cfg.plot_title = "test"
    cfg.plot_fig_size = (6, 6)
    cfg.bbox = None

    with pytest.raises(ValueError, match="iterations must be >= 1"):
        cfg.validate()


def test_gapfill_validate_raises_for_bad_bbox_type(tmp_path: Path) -> None:
    cfg = GapfillRunConfig.__new__(GapfillRunConfig)
    cfg.input_dir = tmp_path
    cfg.save_dir = tmp_path / "out"
    cfg.save_name = "test"
    cfg.num_steps = 100
    cfg.iterations = 5
    cfg.show_plot = False
    cfg.plot_title = "test"
    cfg.plot_fig_size = (6, 6)
    cfg.bbox = [1, 2, 3, 4]  # not a tensor

    with pytest.raises(TypeError, match="bbox must be a torch.Tensor or None"):
        cfg.validate()


def test_gapfill_accepts_tensor_bbox(tmp_path: Path) -> None:
    cfg = GapfillRunConfig(
        input_dir=tmp_path,
        save_dir=tmp_path / "out",
        bbox=torch.tensor([0.1, 0.2, 0.8, 0.9]),
    )
    assert isinstance(cfg.bbox, torch.Tensor)


def test_gapfill_creates_save_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "out"
    GapfillRunConfig(input_dir=tmp_path, save_dir=out)
    assert out.exists()


# ── OutpaintRunConfig ───────────────────────────────────────────────


def test_outpaint_validate_raises_for_missing_input_dir(tmp_path: Path) -> None:
    cfg = OutpaintRunConfig(
        input_dir=tmp_path / "nonexistent",
        save_dir=tmp_path / "out",
    )
    with pytest.raises(FileNotFoundError, match="input_dir does not exist"):
        cfg.validate()


def test_outpaint_validate_raises_for_bad_crop_ratio(tmp_path: Path) -> None:
    cfg = OutpaintRunConfig(
        input_dir=tmp_path,
        save_dir=tmp_path / "out",
        crop_ratio=0.0,
    )
    with pytest.raises(ValueError, match="crop_ratio must be in"):
        cfg.validate()


def test_outpaint_validate_raises_for_bad_direction(tmp_path: Path) -> None:
    cfg = OutpaintRunConfig(
        input_dir=tmp_path,
        save_dir=tmp_path / "out",
        direction="diagonal",
    )
    with pytest.raises(ValueError, match="direction must be one of"):
        cfg.validate()


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_outpaint_accepts_valid_directions(tmp_path: Path, direction: str) -> None:
    cfg = OutpaintRunConfig(
        input_dir=tmp_path,
        save_dir=tmp_path / "out",
        direction=direction,
    )
    cfg.validate()  # should not raise


def test_outpaint_post_init_sets_default_title() -> None:
    cfg = OutpaintRunConfig(
        input_dir=".",
        save_dir=".",
        iterations=5,
        num_steps=100,
    )
    assert "5 iterations" in cfg.plot_title
    assert "100 num_steps" in cfg.plot_title


# ── ThreeDimImputationRunConfig ─────────────────────────────────────


def test_three_dim_validate_raises_for_missing_prev_img(tmp_path: Path) -> None:
    next_img = tmp_path / "next.png"
    next_img.write_bytes(b"fake")

    cfg = ThreeDimImputationRunConfig(
        prev_img_path=tmp_path / "nonexistent.png",
        next_img_path=next_img,
        out_dir=tmp_path / "out",
    )
    with pytest.raises(FileNotFoundError, match="prev_img_path does not exist"):
        cfg.validate()


def test_three_dim_validate_raises_for_bad_weights(tmp_path: Path) -> None:
    prev_img = tmp_path / "prev.png"
    next_img = tmp_path / "next.png"
    prev_img.write_bytes(b"fake")
    next_img.write_bytes(b"fake")

    cfg = ThreeDimImputationRunConfig(
        prev_img_path=prev_img,
        next_img_path=next_img,
        out_dir=tmp_path / "out",
        w_prev=1.5,
    )
    with pytest.raises(ValueError, match="w_prev must be between 0 and 1"):
        cfg.validate()


def test_three_dim_validate_raises_for_empty_save_name(tmp_path: Path) -> None:
    prev_img = tmp_path / "prev.png"
    next_img = tmp_path / "next.png"
    prev_img.write_bytes(b"fake")
    next_img.write_bytes(b"fake")

    cfg = ThreeDimImputationRunConfig(
        prev_img_path=prev_img,
        next_img_path=next_img,
        out_dir=tmp_path / "out",
        save_latents_name="   ",
    )
    with pytest.raises(ValueError, match="save_latents_name must be non-empty"):
        cfg.validate()


# ── ThreeDimImputationWeightSweepConfig ─────────────────────────────


def test_weight_sweep_raises_for_start_greater_than_end(tmp_path: Path) -> None:
    prev_img = tmp_path / "prev.png"
    next_img = tmp_path / "next.png"
    prev_img.write_bytes(b"fake")
    next_img.write_bytes(b"fake")

    cfg = ThreeDimImputationWeightSweepConfig(
        prev_img_path=prev_img,
        next_img_path=next_img,
        out_dir=tmp_path / "out",
        start=0.9,
        end=0.1,
    )
    with pytest.raises(ValueError, match="start must be <= end"):
        cfg.validate()


def test_weight_sweep_raises_when_step_exceeds_range(tmp_path: Path) -> None:
    prev_img = tmp_path / "prev.png"
    next_img = tmp_path / "next.png"
    prev_img.write_bytes(b"fake")
    next_img.write_bytes(b"fake")

    cfg = ThreeDimImputationWeightSweepConfig(
        prev_img_path=prev_img,
        next_img_path=next_img,
        out_dir=tmp_path / "out",
        start=0.4,
        end=0.5,
        step=0.2,
    )
    with pytest.raises(ValueError, match="step is larger than the sweep range"):
        cfg.validate()


def test_weight_sweep_raises_for_zero_step(tmp_path: Path) -> None:
    prev_img = tmp_path / "prev.png"
    next_img = tmp_path / "next.png"
    prev_img.write_bytes(b"fake")
    next_img.write_bytes(b"fake")

    cfg = ThreeDimImputationWeightSweepConfig(
        prev_img_path=prev_img,
        next_img_path=next_img,
        out_dir=tmp_path / "out",
        step=0.0,
    )
    with pytest.raises(ValueError, match="step must be in"):
        cfg.validate()


def test_weight_sweep_valid_config(tmp_path: Path) -> None:
    prev_img = tmp_path / "prev.png"
    next_img = tmp_path / "next.png"
    prev_img.write_bytes(b"fake")
    next_img.write_bytes(b"fake")

    cfg = ThreeDimImputationWeightSweepConfig(
        prev_img_path=prev_img,
        next_img_path=next_img,
        out_dir=tmp_path / "out",
        start=0.1,
        end=0.9,
        step=0.1,
    )
    cfg.validate()  # should not raise
    assert (tmp_path / "out").exists()
