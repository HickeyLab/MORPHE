"""Smoke tests: verify all public modules import without error."""
from __future__ import annotations

import pytest


def test_import_autoencoder_module() -> None:
    from core.autoencoder import (
        Autoencoder,
        AutoencoderArtifact,
        AutoencoderTrainerConfig,
        AutoencoderTrainer,
        AutoencoderInferencer,
    )


def test_import_gcnn_module() -> None:
    from core.gcnn import (
        GCNClassifier,
        GCNNArtifact,
        GCNNTrainerConfig,
        GCNNTrainer,
        GCNNInferencer,
        RegionGraphDataset,
    )


def test_import_latent_diffusion_module() -> None:
    from core.latent_diffusion import (
        LatentArchitectureSpec,
        LatentDiffusionArtifact,
        LatentTrainerConfig,
        LatentDiffusionTrainer,
        LatentTrainTask,
        InpaintTrainTask,
        OutpaintTrainTask,
        ThreeDimImputationTrainTask,
        BaseLatentInferencer,
        InpaintInferencer,
        GapfillInferencer,
        OutpaintInferencer,
        ThreeDimImputationInferencer,
        InpaintRunConfig,
        GapfillRunConfig,
        OutpaintRunConfig,
        ThreeDimImputationRunConfig,
        ThreeDimImputationWeightSweepConfig,
    )


def test_import_pixel_diffusion_module() -> None:
    from core.pixel_diffusion import (
        PixelDiffusionArtifact,
        PixelDiffusionTrainerConfig,
        PrecomputedCascadeDataset,
        PixelDiffusionEvaluator,
        PixelDiffusionInferencer,
        PixelPrecomputeTask,
        InpaintPrecomputeTask,
        OutpaintPrecomputeTask,
        ThreeDimImputationPrecomputeTask,
        PixelDatasetPrecomputer,
        PixelDiffusionTrainer,
    )


def test_import_pipeline_module() -> None:
    from core.pipeline import (
        MorpheArtifact,
        MorpheInferencer,
        MorpheTrainer,
    )


def test_import_preprocessor() -> None:
    from core.preprocessor.config import PreProcessConfig
    from core.preprocessor.preprocess import PreProcessor


def test_import_registry() -> None:
    from core.registry import (
        LATENT_INFERENCER_REGISTRY,
        LATENT_TRAIN_TASK_REGISTRY,
        PRECOMPUTE_TASK_REGISTRY,
    )


def test_import_constants() -> None:
    from constants import InferenceMode

    assert hasattr(InferenceMode, "GAPFILL")
    assert hasattr(InferenceMode, "INPAINTING")
    assert hasattr(InferenceMode, "OUTPAINTING")
    assert hasattr(InferenceMode, "THREE_DIMENSIONAL_IMPUTATION")
