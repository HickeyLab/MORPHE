from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from disco.config import LATENT_TRAIN_STRATEGY_REGISTRY, PreProcessConfig
from disco.core.autoencoder.artifact import AutoencoderArtifact
from disco.core.autoencoder.config import AutoencoderTrainerConfig
from disco.core.autoencoder.inferencer import AutoencoderRGBInferencer
from disco.core.autoencoder.trainer import AutoencoderTrainer
from disco.core.gcnn.artifact import GCNNArtifact
from disco.core.gcnn.config import GCNNTrainerConfig
from disco.core.gcnn.inferencer import GCNNInferencer
from disco.core.gcnn.trainer import GCNNTrainer
from disco.core.latent_diffusion.artifact import LatentDiffusionArtifact
from disco.core.latent_diffusion.infer.run_config import InferenceMode
from disco.core.latent_diffusion.train.base import LatentTrainStrategy
from disco.core.latent_diffusion.train.diffusion_trainer import DiffusionTrainer
from disco.core.latent_diffusion.train.train_config import (
    LatentTrainerConfig,
)
from disco.core.pixel_diffusion.artifact import PixelDiffusionArtifact
from disco.core.pixel_diffusion.config import Cascade512TrainerConfig
from disco.core.pixel_diffusion.precompute.base import PixelPrecomputeStrategy
from disco.core.pixel_diffusion.precompute.pixel_dataset_precomputer import (
    PixelDatasetPrecomputer,
)
from disco.core.pixel_diffusion.precompute.precompute_config import (
    EXPECTED_PRECOMPUTE_STRATEGY_BY_MODE,
)
from disco.core.pixel_diffusion.trainer import Cascade512Trainer
from disco.core.preprocess import PreProcessor

from .artifact import DiscoArtifact


class DiscoTrainer:
    """
    Explicit training entry point for the full DISCO pipeline.
    """

    @staticmethod
    def _apply_seed(seed: int | None) -> None:
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def fit(
        df: pd.DataFrame,
        root_dir: str | Path,
        feature_cols: Sequence[str],
        inference_mode: InferenceMode,
        *,
        device: str | torch.device | None = None,
        seed: int | None = None,
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
        precompute_batch_size: int = 1,
        precompute_num_workers: int = 4,
    ) -> DiscoArtifact:
        if inference_mode is None:
            raise ValueError("Must provide inference_mode.")

        if root_dir is None or str(root_dir).strip() == "":
            raise ValueError("Must provide a root directory for saving and reading files.")
        root_dir = Path(root_dir)

        DiscoTrainer._apply_seed(seed)

        resolved_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        
        preprocess_config = preprocess_config or PreProcessConfig()
        preprocessor = PreProcessor(config=preprocess_config)
        processed_df, _ = preprocessor.preprocess(df=df)

        if gcnn_artifact is None:
            gcnn_trainer = GCNNTrainer(
                df=processed_df,
                cfg=gcnn_trainer_config or GCNNTrainerConfig(),
                feature_cols=feature_cols,
            )
            gcnn_artifact = gcnn_trainer.train()

        gcnn_inferencer = GCNNInferencer.from_artifact(gcnn_artifact)
        processed_df = gcnn_inferencer.predict_proba(processed_df)

        if autoencoder_artifact is None:
            autoencoder_trainer = AutoencoderTrainer(
                df=processed_df,
                cfg=autoencoder_trainer_config or AutoencoderTrainerConfig(),
            )
            autoencoder_artifact = autoencoder_trainer.train()

        autoencoder_inferencer = AutoencoderRGBInferencer.from_artifact(
            autoencoder_artifact
        )
        processed_df = autoencoder_inferencer.add_rgb_and_rasterize_per_region(
            processed_df,
            root_dir,
        )

        expected_latent_training_strategy_cls = LATENT_TRAIN_STRATEGY_REGISTRY[inference_mode]
        resolved_latent_training_strategy = (
            latent_training_strategy
            if latent_training_strategy is not None
            else expected_latent_training_strategy_cls()
        )

        if not isinstance(
            resolved_latent_training_strategy,
            expected_latent_training_strategy_cls,
        ):
            raise TypeError(
                f"latent_training_strategy must be an instance of "
                f"{expected_latent_training_strategy_cls.__name__} "
                f"for inference_mode={inference_mode}, got "
                f"{type(resolved_latent_training_strategy).__name__}"
            )

        if latent_diffusion_artifact is None:
            latent_diffusion_trainer = DiffusionTrainer(
                train_strategy=resolved_latent_training_strategy,
                root_dir=root_dir,
                cfg=latent_diffusion_trainer_config or LatentTrainerConfig(),
            )
            latent_diffusion_artifact = latent_diffusion_trainer.train()

        expected_precompute_strategy_cls = EXPECTED_PRECOMPUTE_STRATEGY_BY_MODE[
            inference_mode
        ]
        resolved_precompute_strategy = (
            pixel_diffusion_precomputer_strategy
            if pixel_diffusion_precomputer_strategy is not None
            else expected_precompute_strategy_cls()
        )

        if not isinstance(
            resolved_precompute_strategy,
            expected_precompute_strategy_cls,
        ):
            raise TypeError(
                f"precompute_strategy must be an instance of "
                f"{expected_precompute_strategy_cls.__name__} "
                f"for inference_mode={inference_mode}, got "
                f"{type(resolved_precompute_strategy).__name__}"
            )

        pd_precomputer = PixelDatasetPrecomputer.from_pretrained(
            vae_path="runwayml/stable-diffusion-v1-5",
            precompute_strategy=resolved_precompute_strategy,
            device=resolved_device,
        )
        train_index, val_index = pd_precomputer.run(
            root_dir=root_dir,
            out_dir=root_dir,
            batch_size=precompute_batch_size,
            num_workers=precompute_num_workers,
        )

        if pixel_diffusion_artifact is None:
            pixel_diffusion_trainer = Cascade512Trainer(
                train_index=Path(train_index),
                val_index=Path(val_index),
                cfg=pixel_diffusion_trainer_config or Cascade512TrainerConfig(),
            )
            pixel_diffusion_artifact = pixel_diffusion_trainer.train()

        return DiscoArtifact(
            preprocessor_config=preprocess_config,
            autoencoder_artifact=autoencoder_artifact,
            gcnn_artifact=gcnn_artifact,
            latent_diffusion_artifact=latent_diffusion_artifact,
            pixel_diffusion_artifact=pixel_diffusion_artifact,
        )