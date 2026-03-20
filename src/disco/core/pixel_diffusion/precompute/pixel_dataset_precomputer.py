from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from disco.core.latent_diffusion.model import VAEEncoder
from disco.core.pixel_diffusion.dataset import PrecomputedCascadeDataset
from disco.core.pixel_diffusion.precompute.base import PixelPrecomputeStrategy

class PixelDatasetPrecomputer:
    def __init__(
        self,
        *,
        vae_encoder: VAEEncoder,
        precompute_strategy: PixelPrecomputeStrategy,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.vae_encoder = vae_encoder.to(self.device).eval()
        self.precompute_strategy = precompute_strategy
        
    @classmethod
    def from_pretrained(
        cls,
        *,
        vae_path: str | Path = "runwayml/stable-diffusion-v1-5",
        precompute_strategy: PixelPrecomputeStrategy,
        device: str | torch.device | None = None,
    ) -> "PixelDatasetPrecomputer":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        vae_encoder = VAEEncoder(pretrained_path=str(vae_path), device=device)
        return cls(
            vae_encoder=vae_encoder,
            precompute_strategy=precompute_strategy,
            device=device,
        )

    @torch.no_grad()
    def run(
        self,
        *,
        root_dir: str | Path,
        out_dir: str | Path,
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> tuple[Path, Path]:
        root_dir = Path(root_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_ds, val_ds = self.precompute_strategy.build_dataset(root_dir=root_dir)

        train_index_path = self._run_split(
            dataset=train_ds,
            split_name="train",
            out_dir=out_dir / "train",
            batch_size=batch_size,
            num_workers=num_workers,
        )

        val_index_path = self._run_split(
            dataset=val_ds,
            split_name="val",
            out_dir=out_dir / "val",
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return train_index_path, val_index_path
    
    def _toCPU16(self, x: torch.Tensor):
        return x.detach().cpu().half()

    @torch.no_grad()
    def _run_split(
        self,
        *,
        dataset: Dataset,
        split_name: str,
        out_dir: Path,
        batch_size: int,
        num_workers: int,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        global_idx = 0
        index_path = out_dir / f"{split_name}_index.jsonl"

        with open(index_path, "w") as fw:
            pbar = tqdm(dl, desc=f"Precomputing {split_name}")

            for batch in pbar:
                encoder_input = self.precompute_strategy.get_encoder_input(batch)
                target_img = self.precompute_strategy.get_target_img(batch)

                encoder_input = encoder_input.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type):
                    z_cond = self.vae_encoder.encode(encoder_input)

                B = encoder_input.size(0)
                for b in range(B):
                    sample_name = self.precompute_strategy.get_sample_name(
                        dataset=dataset,
                        batch=batch,
                        batch_idx=b,
                        global_idx=global_idx + b,
                        split_name=split_name,
                    )

                    metadata = self.precompute_strategy.get_metadata(
                        dataset=dataset,
                        batch=batch,
                        batch_idx=b,
                        global_idx=global_idx + b,
                        split_name=split_name,
                    )

                    pt_path = out_dir / f"{sample_name}.pt"

                    torch.save(
                        {
                            "z_cond": self._toCPU16(z_cond[b]),
                            "target_img": self._toCPU16(target_img[b]),
                            **metadata,
                        },
                        pt_path,
                    )

                    fw.write(json.dumps({"pt": str(pt_path)}) + "\n")

                global_idx += B

                if global_idx % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

        return index_path