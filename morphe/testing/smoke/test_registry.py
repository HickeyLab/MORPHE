"""Smoke tests: verify registries are populated and consistent."""
from __future__ import annotations

from constants import InferenceMode
from core.registry import (
    LATENT_INFERENCER_REGISTRY,
    LATENT_TRAIN_TASK_REGISTRY,
    PRECOMPUTE_TASK_REGISTRY,
)


def test_all_registries_non_empty() -> None:
    assert len(LATENT_INFERENCER_REGISTRY) > 0
    assert len(LATENT_TRAIN_TASK_REGISTRY) > 0
    assert len(PRECOMPUTE_TASK_REGISTRY) > 0


def test_registries_have_matching_keys() -> None:
    assert set(LATENT_INFERENCER_REGISTRY) == set(LATENT_TRAIN_TASK_REGISTRY)
    assert set(LATENT_INFERENCER_REGISTRY) == set(PRECOMPUTE_TASK_REGISTRY)


def test_all_inference_modes_covered() -> None:
    all_modes = set(InferenceMode)
    registered = set(LATENT_INFERENCER_REGISTRY)
    assert registered == all_modes


def test_registry_values_are_classes() -> None:
    for cls in LATENT_INFERENCER_REGISTRY.values():
        assert isinstance(cls, type)
    for cls in LATENT_TRAIN_TASK_REGISTRY.values():
        assert isinstance(cls, type)
    for cls in PRECOMPUTE_TASK_REGISTRY.values():
        assert isinstance(cls, type)
