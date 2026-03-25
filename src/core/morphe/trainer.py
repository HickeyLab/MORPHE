from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence, TypeVar

import numpy as np
import pandas as pd
import torch

from core.autoencoder.artifact import AutoencoderArtifact
from core.autoencoder.config import AutoencoderTrainerConfig
from core.autoencoder.inferencer import AutoencoderInferencer
from core.autoencoder.trainer import AutoencoderTrainer
from core.gcnn.artifact import GCNNArtifact
from core.gcnn.config import GCNNTrainerConfig
from core.gcnn.inferencer import GCNNInferencer
from core.gcnn.trainer import GCNNTrainer
from core.latent_diffusion.artifact import LatentDiffusionArtifact
from core.latent_diffusion.train.base import LatentTrainStrategy
from core.latent_diffusion.train.train_config import LatentTrainerConfig
from core.latent_diffusion.train.trainer import DiffusionTrainer
from core.pixel_diffusion.artifact import PixelDiffusionArtifact
from core.pixel_diffusion.config import Cascade512TrainerConfig
from core.pixel_diffusion.precompute.base import PixelPrecomputeStrategy
from core.pixel_diffusion.precompute.pixel_dataset_precomputer import (
    PixelDatasetPrecomputer,
)
from core.pixel_diffusion.trainer import Cascade512Trainer
from core.preprocessor.config import PreProcessConfig
from core.preprocessor.preprocess import PreProcessor
from core.registry import (
    LATENT_TRAIN_STRATEGY_REGISTRY,
    PRECOMPUTE_STRATEGY_REGISTRY,
    InferenceMode,
)
from utils import resolve_device, resolve_dtype

from .artifact import MorpheArtifact

T = TypeVar("T")


class MorpheTrainer:
    """
    Explicit training entry point for the full MORPHE pipeline.
    """

    @staticmethod
    def _apply_seed(seed: int | None) -> None:
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_strategy(
        *,
        provided: T | None,
        expected_cls: type[T],
        arg_name: str,
        inference_mode: InferenceMode,
    ) -> T:
        resolved = provided if provided is not None else expected_cls()

        if not isinstance(resolved, expected_cls):
            raise TypeError(
                f"{arg_name} must be an instance of {expected_cls.__name__} "
                f"for inference_mode={inference_mode}, got "
                f"{type(resolved).__name__}"
            )

        return resolved

    @staticmethod
    def train(
        df: pd.DataFrame,
        root_dir: str | Path,
        feature_cols: Sequence[str],
        inference_mode: InferenceMode,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        seed: int | None = None,
        verbose: bool = False,
        preprocess_config: PreProcessConfig | None = None,
        autoencoder_trainer_config: AutoencoderTrainerConfig | None = None,
        gcnn_trainer_config: GCNNTrainerConfig | None = None,
        latent_diffusion_trainer_config: LatentTrainerConfig | None = None,
        latent_training_strategy: LatentTrainStrategy | None = None,
        pixel_diffusion_precomputer_strategy: PixelPrecomputeStrategy | None = None,
        pixel_diffusion_trainer_config: Cascade512TrainerConfig | None = None,
        autoencoder_artifact: AutoencoderArtifact | None = None,
        gcnn_artifact: GCNNArtifact | None = None,
        latent_diffusion_artifact: LatentDiffusionArtifact | None = None,
        pixel_diffusion_artifact: PixelDiffusionArtifact | None = None,
        ld_trainer_save_best: bool = True,
        ld_trainer_save_last: bool = True,
        ld_trainer_checkpoint_every_epochs: int | None = None,
        ld_trainer_save_name: str = "latent_diffuser_artifact.pt",
        pd_precomputer_root_dir: str | Path | None = None,
        pd_precomputer_out_dir: str | Path | None = None,
        pd_precomputer_batch_size: int = 1,
        pd_precomputer_num_workers: int = 4,
        pd_trainer_enable_epoch_visualizations: bool = False,
        pd_trainer_epoch_visualization_max_batches: int = 2,
        pd_trainer_enable_composition_eval: bool = False,
        pd_trainer_chart_left_title: str = "Original",
        pd_trainer_chart_right_title: str = "Predicted",
    ) -> MorpheArtifact:
        if inference_mode is None:
            raise ValueError("Must provide inference_mode.")

        if root_dir is None or str(root_dir).strip() == "":
            raise ValueError(
                "Must provide a root directory for saving and reading files."
            )

        root_dir = Path(root_dir)
        MorpheTrainer._apply_seed(seed)
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(resolved_device, dtype)

        if verbose:
            print(
                f"[MorpheTrainer] Starting training pipeline "
                f"(device={resolved_device}, dtype={resolved_dtype}, seed={seed})."
            )

        preprocess_config = preprocess_config or PreProcessConfig()
        processed_df, _ = PreProcessor(config=preprocess_config).preprocess(
            df=df, verbose=verbose
        )

        if gcnn_artifact is None:
            if verbose:
                print("[MorpheTrainer] Training GCNN component.")

            gcnn_trainer = GCNNTrainer(
                df=processed_df,
                cfg=gcnn_trainer_config or GCNNTrainerConfig(),
                feature_cols=feature_cols,
                device=resolved_device,
                seed=seed,
            )
            gcnn_artifact = gcnn_trainer.train(verbose=verbose)
        elif verbose:
            print("[MorpheTrainer] Using provided GCNN artifact.")

        gcnn_inferencer = GCNNInferencer.from_artifact(
            artifact=gcnn_artifact,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        processed_df = gcnn_inferencer.predict_proba(processed_df)

        if autoencoder_artifact is None:
            if verbose:
                print("[MorpheTrainer] Training autoencoder component.")

            autoencoder_trainer = AutoencoderTrainer(
                df=processed_df,
                cfg=autoencoder_trainer_config or AutoencoderTrainerConfig(),
                device=resolved_device,
                seed=seed,
            )
            autoencoder_artifact = autoencoder_trainer.train(verbose=verbose)
        elif verbose:
            print("[MorpheTrainer] Using provided autoencoder artifact.")

        autoencoder_inferencer = AutoencoderInferencer.from_artifact(
            artifact=autoencoder_artifact,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        processed_df = autoencoder_inferencer.add_rgb_and_rasterize_per_region(
            processed_df,
            root_dir,
        )

        expected_latent_strategy_cls = LATENT_TRAIN_STRATEGY_REGISTRY[inference_mode]
        resolved_latent_strategy = MorpheTrainer._resolve_strategy(
            provided=latent_training_strategy,
            expected_cls=expected_latent_strategy_cls,
            arg_name="latent_training_strategy",
            inference_mode=inference_mode,
        )

        if latent_diffusion_artifact is None:
            if verbose:
                print("[MorpheTrainer] Training latent diffusion component.")

            latent_diffusion_trainer = DiffusionTrainer(
                train_strategy=resolved_latent_strategy,
                root_dir=root_dir,
                cfg=latent_diffusion_trainer_config or LatentTrainerConfig(),
                device=resolved_device,
                dtype=resolved_dtype,
                seed=seed,
            )
            latent_train_result = latent_diffusion_trainer.train(
                save_best=ld_trainer_save_best,
                save_last=ld_trainer_save_last,
                checkpoint_every_epochs=ld_trainer_checkpoint_every_epochs,
                save_name=ld_trainer_save_name,
                verbose=verbose,
            )
            latent_diffusion_artifact = latent_train_result.artifact
        elif verbose:
            print("[MorpheTrainer] Using provided latent diffusion artifact.")

        if pixel_diffusion_artifact is None:
            expected_precompute_strategy_cls = PRECOMPUTE_STRATEGY_REGISTRY[
                inference_mode
            ]
            resolved_precompute_strategy = MorpheTrainer._resolve_strategy(
                provided=pixel_diffusion_precomputer_strategy,
                expected_cls=expected_precompute_strategy_cls,
                arg_name="pixel_diffusion_precomputer_strategy",
                inference_mode=inference_mode,
            )

            if verbose:
                print("[MorpheTrainer] Precomputing pixel diffusion dataset.")

            pd_precomputer = PixelDatasetPrecomputer.from_pretrained(
                precompute_strategy=resolved_precompute_strategy,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            train_index, val_index = pd_precomputer.precompute(
                root_dir=(
                    Path(pd_precomputer_root_dir)
                    if pd_precomputer_root_dir
                    else root_dir
                ),
                out_dir=(
                    Path(pd_precomputer_out_dir) if pd_precomputer_out_dir else root_dir
                ),
                batch_size=pd_precomputer_batch_size,
                num_workers=pd_precomputer_num_workers,
            )

            if verbose:
                print("[MorpheTrainer] Training pixel diffusion component.")

            pixel_diffusion_trainer = Cascade512Trainer(
                train_index=Path(train_index),
                val_index=Path(val_index),
                cfg=pixel_diffusion_trainer_config or Cascade512TrainerConfig(),
                device=resolved_device,
                seed=seed,
                verbose=verbose,
            )
            pixel_diffusion_artifact = pixel_diffusion_trainer.train(
                enable_epoch_visualizations=pd_trainer_enable_epoch_visualizations,
                epoch_visualization_max_batches=pd_trainer_epoch_visualization_max_batches,
                enable_composition_eval=pd_trainer_enable_composition_eval,
                chart_left_title=pd_trainer_chart_left_title,
                chart_right_title=pd_trainer_chart_right_title,
                verbose=verbose,
            )
        elif verbose:
            print("[MorpheTrainer] Using provided pixel diffusion artifact.")

        if verbose:
            print("[MorpheTrainer] Finished training pipeline.")

        return MorpheArtifact(
            preprocessor_config=preprocess_config,
            autoencoder_artifact=autoencoder_artifact,
            gcnn_artifact=gcnn_artifact,
            latent_diffusion_artifact=latent_diffusion_artifact,
            pixel_diffusion_artifact=pixel_diffusion_artifact,
        )
