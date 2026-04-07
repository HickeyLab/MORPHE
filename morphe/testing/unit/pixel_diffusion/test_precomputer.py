from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from torch.utils.data import Dataset

from core.pixel_diffusion.precompute.pixel_dataset_precomputer import PixelDatasetPrecomputer
from core.pixel_diffusion.precompute.base import PixelPrecomputeTask


# ── Dummy helpers ───────────────────────────────────────────────────


class DummyDataset(Dataset):
    """Minimal dataset returning a dict with cond_img and target_img."""

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.filenames = [f"sample_{i}" for i in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "cond_img": torch.randn(3, 64, 64),
            "target_img": torch.randn(3, 512, 512),
            "filename": self.filenames[idx],
        }


class DummyPrecomputeTask(PixelPrecomputeTask):
    ae_pretrained_path: str = "fake/path"

    def build_dataset(self, root_dir: Path) -> tuple[DummyDataset, DummyDataset]:
        return DummyDataset(size=3), DummyDataset(size=2)

    def get_encoder_input(self, batch) -> torch.Tensor:
        return batch["cond_img"]

    def get_target_img(self, batch) -> torch.Tensor:
        return batch["target_img"]

    def get_sample_name(
        self,
        *,
        dataset,
        batch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> str:
        return f"{split_name}_{global_idx:04d}"

    def get_metadata(
        self,
        *,
        dataset,
        batch,
        batch_idx: int,
        global_idx: int,
        split_name: Literal["train", "val"],
    ) -> dict[str, object]:
        return {"filename": batch["filename"][batch_idx]}


class DummyVAEEncoder(torch.nn.Module):
    """Fake VAE that returns a fixed-shape latent."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        B = imgs.size(0)
        return torch.randn(B, 4, 8, 8)


# ── precompute validation ───────────────────────────────────────────


def test_precompute_raises_for_missing_root_dir(tmp_path: Path) -> None:
    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="root_dir does not exist"):
        precomputer.precompute(
            root_dir=tmp_path / "nonexistent",
            out_dir=tmp_path / "out",
        )


def test_precompute_raises_for_invalid_batch_size(tmp_path: Path) -> None:
    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        precomputer.precompute(
            root_dir=tmp_path,
            out_dir=tmp_path / "out",
            batch_size=0,
        )


def test_precompute_raises_for_negative_num_workers(tmp_path: Path) -> None:
    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    with pytest.raises(ValueError, match="num_workers must be non-negative"):
        precomputer.precompute(
            root_dir=tmp_path,
            out_dir=tmp_path / "out",
            num_workers=-1,
        )


# ── precompute end-to-end ───────────────────────────────────────────


def test_precompute_writes_pt_files_and_index(tmp_path: Path) -> None:
    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    train_idx, val_idx = precomputer.precompute(
        root_dir=tmp_path,
        out_dir=tmp_path / "out",
        batch_size=2,
        num_workers=0,
    )

    # Index files exist
    assert train_idx.exists()
    assert val_idx.exists()

    # Train split: 3 samples -> 3 .pt files
    train_dir = tmp_path / "out" / "train"
    train_pt_files = list(train_dir.glob("*.pt"))
    assert len(train_pt_files) == 3

    # Val split: 2 samples -> 2 .pt files
    val_dir = tmp_path / "out" / "val"
    val_pt_files = list(val_dir.glob("*.pt"))
    assert len(val_pt_files) == 2

    # Index file has correct number of lines
    with train_idx.open() as f:
        train_lines = f.readlines()
    assert len(train_lines) == 3

    with val_idx.open() as f:
        val_lines = f.readlines()
    assert len(val_lines) == 2


def test_precompute_index_contains_valid_json(tmp_path: Path) -> None:
    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    train_idx, _ = precomputer.precompute(
        root_dir=tmp_path,
        out_dir=tmp_path / "out",
        batch_size=1,
        num_workers=0,
    )

    with train_idx.open() as f:
        for line in f:
            record = json.loads(line)
            assert "pt" in record
            assert Path(record["pt"]).exists()


def test_precompute_pt_files_contain_expected_keys(tmp_path: Path) -> None:
    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    precomputer.precompute(
        root_dir=tmp_path,
        out_dir=tmp_path / "out",
        batch_size=1,
        num_workers=0,
    )

    pt_files = list((tmp_path / "out" / "train").glob("*.pt"))
    assert len(pt_files) > 0

    sample = torch.load(pt_files[0], map_location="cpu")
    assert "z_cond" in sample
    assert "target_img" in sample
    assert "filename" in sample


def test_precompute_creates_output_dirs(tmp_path: Path) -> None:
    out_dir = tmp_path / "deep" / "nested" / "out"

    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    precomputer.precompute(
        root_dir=tmp_path,
        out_dir=out_dir,
        batch_size=1,
        num_workers=0,
    )

    assert (out_dir / "train").exists()
    assert (out_dir / "val").exists()


# ── _to_cpu_half ────────────────────────────────────────────────────


def test_to_cpu_half_detaches_and_moves_to_cpu() -> None:
    precomputer = PixelDatasetPrecomputer(
        vae_encoder=DummyVAEEncoder(),
        precompute_task=DummyPrecomputeTask(),
        device="cpu",
    )

    t = torch.randn(2, 3, requires_grad=True)
    result = precomputer._to_cpu_half(t)

    assert result.device.type == "cpu"
    assert not result.requires_grad
