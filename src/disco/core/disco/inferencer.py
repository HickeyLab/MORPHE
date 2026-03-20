from __future__ import annotations

from pathlib import Path
from re import I
from typing import Any

import pandas as pd
import torch
from disco.config import LATENT_INFERENCER_REGISTRY, LATENT_TRAIN_STRATEGY_REGISTRY, InferenceMode, PreProcessConfig

from disco.core.autoencoder.artifact import AutoencoderArtifact
from disco.core.autoencoder.inferencer import AutoencoderRGBInferencer
from disco.core.disco.artifact import DiscoArtifact
from disco.core.gcnn.artifact import GCNNArtifact
from disco.core.gcnn.inferencer import GCNNInferencer
from disco.core.latent_diffusion.artifact import LatentDiffusionArtifact
from disco.core.latent_diffusion.infer.base import BaseLatentInferencer
from disco.core.latent_diffusion.infer.run_config import GapfillRunConfig, InpaintRunConfig, LatentBaseRunConfig, OutpaintRunConfig, ThreeDimImputationRunConfig
from disco.core.latent_diffusion.infer.three_dim import ThreeDimImputationInferencer
from disco.core.pixel_diffusion.artifact import PixelDiffusionArtifact
from disco.core.pixel_diffusion.inferencer import PixelDiffusionInferencer
from disco.core.preprocess import PreProcessor


class DiscoInferencer:
    """
    High-level inference orchestrator.

    Delegates generation to latent diffusion and decoding/refinement
    to pixel diffusion or autoencoder.
    """

    def __init__(
        self,
        *,
        artifact: DiscoArtifact | None = None,
        preprocessor_config: PreProcessConfig | None = None,
        ae_artifact: AutoencoderArtifact | None = None,
        gcnn_artifact: GCNNArtifact | None = None,
        ld_artifact: LatentDiffusionArtifact | None = None,
        pd_artifact: PixelDiffusionArtifact | None = None,
        pd_pretrained_path: str = "runwayml/stable-diffusion-v1-5",
        pd_num_inference_steps: int = 150,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)

        if artifact is not None:
            preprocessor_config = preprocessor_config or artifact.preprocessor_config
            ae_artifact = ae_artifact or artifact.autoencoder_artifact
            gcnn_artifact = gcnn_artifact or artifact.gcnn_artifact
            ld_artifact = ld_artifact or artifact.latent_diffusion_artifact
            pd_artifact = pd_artifact or artifact.pixel_diffusion_artifact

        if (
            ae_artifact is None
            or gcnn_artifact is None
            or ld_artifact is None
            or pd_artifact is None
        ):
            raise ValueError(
                "All artifacts must be provided either directly or via the DiscoArtifact"
            )

        self.artifact = artifact
        self.preprocessor_config = (
            preprocessor_config
            if preprocessor_config is not None
            else PreProcessConfig()
        )
        self.ae_artifact = ae_artifact
        self.gcnn_artifact = gcnn_artifact
        self.ld_artifact = ld_artifact
        self.pd_artifact = pd_artifact
        self.device = device
        self.dtype = dtype

        self._pp: PreProcessor = PreProcessor(config=self.preprocessor_config)
        self._ae: AutoencoderRGBInferencer = AutoencoderRGBInferencer.from_artifact(
                artifact=self.ae_artifact,
                device=self.device,
            )
        self._gcnn: GCNNInferencer = GCNNInferencer.from_artifact(
                artifact=self.gcnn_artifact,
                device=self.device,
            )
        self._pd: PixelDiffusionInferencer = PixelDiffusionInferencer.from_artifact(
            artifact=self.pd_artifact,
            pretrained_path=pd_pretrained_path,
            num_inference_steps=pd_num_inference_steps,
            device=self.device,
            dtype=self.dtype,
        )
        
    def _build_ld_inferencer(self, mode: InferenceMode) -> BaseLatentInferencer:
        inferencer_cls = LATENT_INFERENCER_REGISTRY.get(mode)
        train_strategy = LATENT_TRAIN_STRATEGY_REGISTRY.get(mode)
        
        if inferencer_cls is None:
            raise ValueError(f"Unsupported inference mode: {mode}")
        
        if train_strategy is None:
            raise ValueError(f"Unsupported train strategy for mode: {mode}")

        return inferencer_cls(
            train_strategy=train_strategy(),
            artifact=self.ld_artifact,
            device=self.device,
            dtype=self.dtype,
        )

    def _run_task(
        self,
        df: pd.DataFrame,
        inference_mode: InferenceMode,
        root_dir: str | Path,
        latent_run_config: LatentBaseRunConfig,
    ) -> list[torch.Tensor]:
        if not root_dir:
            raise ValueError("Must provide root_dir for saving intermediate files.")
        root_dir = Path(root_dir)
        
        processed_df, _ = self._pp.preprocess(df=df)
        
        # GCNN
        gcnn_probabilities = self._gcnn.predict_proba(df=processed_df)
        
        # Autoencoder + Rasterization
        rgb_values = self._ae.add_rgb_and_rasterize_per_region(
            result_df=gcnn_probabilities,
            save_dir=root_dir,
        )
        
        ld = self._build_ld_inferencer(mode=inference_mode)        
        # Latent Diffusion
        latents = ld.run(cfg=latent_run_config)
        
        results = []
        for latent in latents:
            rgb = self._pd.decode(latent=latent)
            cell_map = self._ae.decode_rgb_to_cell_map(rgb)
            results.append(cell_map)

        return results

    def inpaint(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        latent_run_config: InpaintRunConfig | None = None,
    ) -> Any:
        latent_run_config = latent_run_config or InpaintRunConfig(
            image_dir=root_dir,
            mask_dir=root_dir,
        )   
        return self._run_task(
            df=df,
            inference_mode=InferenceMode.INPAINTING,
            root_dir=root_dir,
            latent_run_config=latent_run_config,
        )

    def outpaint(
        self, 
        df: pd.DataFrame, 
        root_dir: str | Path,
        latent_run_config: OutpaintRunConfig | None = None,
    ) -> Any:
        latent_run_config = latent_run_config or OutpaintRunConfig(
            original_dir=root_dir,
            save_dir=root_dir
        )
        
        return self._run_task(
            df=df,
            inference_mode=InferenceMode.OUTPAINTING,
            root_dir=root_dir,
            latent_run_config=latent_run_config,
        )

    def gapfill(
        self, 
        df: pd.DataFrame, 
        root_dir: str | Path,
        latent_run_config: GapfillRunConfig | None = None,
    ) -> Any:
        latent_run_config = latent_run_config or GapfillRunConfig(
            original_dir=root_dir,
            save_dir=root_dir
        )
        
        return self._run_task(
            df=df,
            inference_mode=InferenceMode.GAPFILL,
            root_dir=root_dir,
            latent_run_config=latent_run_config,
        )

    def three_dimensional_imputation(
        self, 
        df: pd.DataFrame, 
        root_dir: str | Path,
        latent_run_config: ThreeDimImputationRunConfig | None = None
    ) -> Any:
        latent_run_config = latent_run_config or ThreeDimImputationRunConfig(
            prev_path=root_dir,
            next_path=root_dir,
            out_dir=root_dir,
        )
        
        return self._run_task(
            df=df,
            inference_mode=InferenceMode.THREE_DIMENSIONAL_IMPUTATION,
            root_dir=root_dir,
            latent_run_config=latent_run_config,
        )
