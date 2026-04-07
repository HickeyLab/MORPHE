from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from core.latent_diffusion.model import VAEEncoder
from core.pixel_diffusion.precompute.base import PixelPrecomputeTask
from utils import resolve_device, resolve_dtype


class PixelDatasetPrecomputer:
    """
    Precomputes pixel diffusion training examples and writes them to disk.

    This utility uses a VAE encoder plus a caller-provided precompute strategy
    to build train/validation datasets, encode conditioning inputs into latent
    representations, and save per-sample `.pt` files alongside JSONL index files.
    """

    def __init__(
        self,
        *,
        vae_encoder: VAEEncoder,
        precompute_task: PixelPrecomputeTask,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the precomputer.

        Args:
            vae_encoder: Encoder used to produce conditioning latents.
            precompute_task: Strategy that defines dataset construction,
                sample extraction, naming, and metadata generation.
            device: Target compute device. If omitted, CUDA is used when
                available; otherwise CPU.
        """
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self.vae_encoder = vae_encoder.to(self.device)
        self.vae_encoder.eval()
        self.precompute_task = precompute_task

    @classmethod
    def from_pretrained(
        cls,
        *,
        precompute_task: PixelPrecomputeTask,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> PixelDatasetPrecomputer:
        """
        Build a precomputer from a pretrained VAE checkpoint/path.

        Args:
            vae_path: Pretrained VAE source path or model identifier.
            precompute_task: Strategy that defines dataset construction,
                sample extraction, naming, and metadata generation.
            device: Target compute device. If omitted, CUDA is used when
                available; otherwise CPU.

        Returns:
            A configured `PixelDatasetPrecomputer`.
        """
        resolved_device = resolve_device(device)
        vae_encoder = VAEEncoder(
            pretrained_path=precompute_task.ae_pretrained_path,
            device=resolved_device,
        )
        return cls(
            vae_encoder=vae_encoder,
            precompute_task=precompute_task,
            device=resolved_device,
            dtype=dtype,
        )

    def _to_cpu_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detach a tensor and move it to CPU in float16 format for compact storage.
        """
        return x.detach().to(device="cpu", dtype=self.dtype)

    @torch.no_grad()
    def precompute(
        self,
        *,
        root_dir: str | Path,
        out_dir: str | Path,
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> tuple[Path, Path]:
        """
        Precompute train and validation splits and write their index files.

        Args:
            root_dir: Root directory used by the strategy to build datasets.
            out_dir: Output directory where split subdirectories and index files
                will be written.
            batch_size: Batch size used during precomputation.
            num_workers: Number of DataLoader worker processes.

        Returns:
            A tuple of `(train_index_path, val_index_path)`.
        """
        root_dir = Path(root_dir)
        out_dir = Path(out_dir)

        if not root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")

        out_dir.mkdir(parents=True, exist_ok=True)

        train_ds, val_ds = self.precompute_task.build_dataset(root_dir=root_dir)

        train_index_path = self._precompute_split(
            dataset=train_ds,
            split_name="train",
            out_dir=out_dir / "train",
            batch_size=batch_size,
            num_workers=num_workers,
        )
        val_index_path = self._precompute_split(
            dataset=val_ds,
            split_name="val",
            out_dir=out_dir / "val",
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return train_index_path, val_index_path

    @torch.no_grad()
    def _precompute_split(
        self,
        *,
        dataset: Dataset,
        split_name: Literal["train", "val"],
        out_dir: Path,
        batch_size: int,
        num_workers: int,
    ) -> Path:
        """
        Precompute one dataset split and write its JSONL index file.

        Args:
            dataset: Dataset for the split being precomputed.
            split_name: Human-readable split name used in filenames and logging.
            out_dir: Directory for the split's output files.
            batch_size: Batch size used during precomputation.
            num_workers: Number of DataLoader worker processes.

        Returns:
            Path to the generated JSONL index file for the split.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        global_idx = 0
        index_path = out_dir / f"{split_name}_index.jsonl"

        with index_path.open("w", encoding="utf-8") as index_file:
            progress_bar = tqdm(dataloader, desc=f"Precomputing {split_name}")

            for batch in progress_bar:
                encoder_input = self.precompute_task.get_encoder_input(batch)
                target_img = self.precompute_task.get_target_img(batch)
                encoder_input = encoder_input.to(self.device, non_blocking=True)

                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        z_cond = self.vae_encoder.encode(encoder_input)
                else:
                    z_cond = self.vae_encoder.encode(encoder_input)

                batch_size_actual = encoder_input.size(0)

                for batch_idx in range(batch_size_actual):
                    sample_name = self.precompute_task.get_sample_name(
                        dataset=dataset,
                        batch=batch,
                        batch_idx=batch_idx,
                        global_idx=global_idx + batch_idx,
                        split_name=split_name,
                    )
                    metadata = self.precompute_task.get_metadata(
                        dataset=dataset,
                        batch=batch,
                        batch_idx=batch_idx,
                        global_idx=global_idx + batch_idx,
                        split_name=split_name,
                    )

                    pt_path = out_dir / f"{sample_name}.pt"
                    torch.save(
                        {
                            "z_cond": self._to_cpu_half(z_cond[batch_idx]),
                            "target_img": self._to_cpu_half(target_img[batch_idx]),
                            **metadata,
                        },
                        pt_path,
                    )

                    record = {"pt": str(pt_path)}
                    index_file.write(json.dumps(record) + "\n")

                global_idx += batch_size_actual

        return index_path