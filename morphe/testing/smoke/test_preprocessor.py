"""Smoke tests: verify preprocessor instantiation and basic usage."""
from __future__ import annotations

import pandas as pd

from core.preprocessor.config import PreProcessConfig
from core.preprocessor.preprocess import PreProcessor


def test_preprocessor_instantiates_with_default_config() -> None:
    config = PreProcessConfig()
    preprocessor = PreProcessor(config=config)
    assert preprocessor.config is config


def test_preprocessor_instantiates_with_custom_config() -> None:
    config = PreProcessConfig(original_dimensions=(1000, 2000))
    preprocessor = PreProcessor(config=config)
    assert preprocessor.config.original_dimensions == (1000, 2000)
