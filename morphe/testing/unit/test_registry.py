from __future__ import annotations

import pytest

from constants import InferenceMode
from core.registry import (
    LATENT_INFERENCER_REGISTRY,
    LATENT_TRAIN_TASK_REGISTRY,
    PRECOMPUTE_TASK_REGISTRY,
)
from core.latent_diffusion.inference.base import BaseLatentInferencer
from core.latent_diffusion.inference.gapfill import GapfillInferencer
from core.latent_diffusion.inference.inpaint import InpaintInferencer
from core.latent_diffusion.inference.outpaint import OutpaintInferencer
from core.latent_diffusion.inference.three_dim import ThreeDimImputationInferencer
from core.latent_diffusion.train.base import LatentTrainTask
from core.latent_diffusion.train.inpaint import InpaintTrainTask
from core.latent_diffusion.train.outpaint import OutpaintTrainTask
from core.latent_diffusion.train.three_dim import ThreeDimImputationTrainTask
from core.pixel_diffusion.precompute.inpaint import InpaintPrecomputeTask
from core.pixel_diffusion.precompute.outpaint import OutpaintPrecomputeTask
from core.pixel_diffusion.precompute.three_dim import ThreeDimImputationPrecomputeTask


# ── LATENT_INFERENCER_REGISTRY ─────────────────────────────────────


class TestLatentInferencerRegistry:
    def test_all_inference_modes_present(self) -> None:
        expected_modes = {
            InferenceMode.GAPFILL,
            InferenceMode.INPAINTING,
            InferenceMode.OUTPAINTING,
            InferenceMode.THREE_DIMENSIONAL_IMPUTATION,
        }
        assert set(LATENT_INFERENCER_REGISTRY.keys()) == expected_modes

    def test_gapfill_maps_to_gapfill_inferencer(self) -> None:
        assert LATENT_INFERENCER_REGISTRY[InferenceMode.GAPFILL] is GapfillInferencer

    def test_inpainting_maps_to_inpaint_inferencer(self) -> None:
        assert LATENT_INFERENCER_REGISTRY[InferenceMode.INPAINTING] is InpaintInferencer

    def test_outpainting_maps_to_outpaint_inferencer(self) -> None:
        assert LATENT_INFERENCER_REGISTRY[InferenceMode.OUTPAINTING] is OutpaintInferencer

    def test_three_dim_maps_to_three_dim_inferencer(self) -> None:
        assert (
            LATENT_INFERENCER_REGISTRY[InferenceMode.THREE_DIMENSIONAL_IMPUTATION]
            is ThreeDimImputationInferencer
        )

    def test_all_values_are_base_inferencer_subclasses(self) -> None:
        for mode, cls in LATENT_INFERENCER_REGISTRY.items():
            assert issubclass(cls, BaseLatentInferencer), (
                f"{mode} maps to {cls}, which is not a BaseLatentInferencer subclass"
            )


# ── LATENT_TRAIN_TASK_REGISTRY ─────────────────────────────────────


class TestLatentTrainTaskRegistry:
    def test_all_inference_modes_present(self) -> None:
        expected_modes = {
            InferenceMode.GAPFILL,
            InferenceMode.INPAINTING,
            InferenceMode.OUTPAINTING,
            InferenceMode.THREE_DIMENSIONAL_IMPUTATION,
        }
        assert set(LATENT_TRAIN_TASK_REGISTRY.keys()) == expected_modes

    def test_inpainting_maps_to_inpaint_train_task(self) -> None:
        assert LATENT_TRAIN_TASK_REGISTRY[InferenceMode.INPAINTING] is InpaintTrainTask

    def test_outpainting_maps_to_outpaint_train_task(self) -> None:
        assert LATENT_TRAIN_TASK_REGISTRY[InferenceMode.OUTPAINTING] is OutpaintTrainTask

    def test_three_dim_maps_to_three_dim_train_task(self) -> None:
        assert (
            LATENT_TRAIN_TASK_REGISTRY[InferenceMode.THREE_DIMENSIONAL_IMPUTATION]
            is ThreeDimImputationTrainTask
        )

    def test_all_values_are_latent_train_task_subclasses(self) -> None:
        for mode, cls in LATENT_TRAIN_TASK_REGISTRY.items():
            assert issubclass(cls, LatentTrainTask), (
                f"{mode} maps to {cls}, which is not a LatentTrainTask subclass"
            )


# ── PRECOMPUTE_TASK_REGISTRY ───────────────────────────────────────


class TestPrecomputeTaskRegistry:
    def test_all_inference_modes_present(self) -> None:
        expected_modes = {
            InferenceMode.GAPFILL,
            InferenceMode.INPAINTING,
            InferenceMode.OUTPAINTING,
            InferenceMode.THREE_DIMENSIONAL_IMPUTATION,
        }
        assert set(PRECOMPUTE_TASK_REGISTRY.keys()) == expected_modes

    def test_inpainting_maps_to_inpaint_precompute_task(self) -> None:
        assert PRECOMPUTE_TASK_REGISTRY[InferenceMode.INPAINTING] is InpaintPrecomputeTask

    def test_outpainting_maps_to_outpaint_precompute_task(self) -> None:
        assert PRECOMPUTE_TASK_REGISTRY[InferenceMode.OUTPAINTING] is OutpaintPrecomputeTask

    def test_three_dim_maps_to_three_dim_precompute_task(self) -> None:
        assert (
            PRECOMPUTE_TASK_REGISTRY[InferenceMode.THREE_DIMENSIONAL_IMPUTATION]
            is ThreeDimImputationPrecomputeTask
        )


# ── cross-registry consistency ─────────────────────────────────────


class TestRegistryConsistency:
    def test_all_registries_cover_same_modes(self) -> None:
        assert set(LATENT_INFERENCER_REGISTRY.keys()) == set(
            LATENT_TRAIN_TASK_REGISTRY.keys()
        )
        assert set(LATENT_INFERENCER_REGISTRY.keys()) == set(
            PRECOMPUTE_TASK_REGISTRY.keys()
        )

    def test_no_none_values_in_any_registry(self) -> None:
        for registry_name, registry in [
            ("LATENT_INFERENCER_REGISTRY", LATENT_INFERENCER_REGISTRY),
            ("LATENT_TRAIN_TASK_REGISTRY", LATENT_TRAIN_TASK_REGISTRY),
            ("PRECOMPUTE_TASK_REGISTRY", PRECOMPUTE_TASK_REGISTRY),
        ]:
            for mode, cls in registry.items():
                assert cls is not None, f"{registry_name}[{mode}] is None"
