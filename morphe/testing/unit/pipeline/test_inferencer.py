from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from core.pipeline.inferencer import MorpheInferencer


# ── helpers ─────────────────────────────────────────────────────────


def _mock(cls_name: str) -> MagicMock:
    """Create a MagicMock with a __class__.__name__ for error messages."""
    m = MagicMock()
    m.__class__ = type(cls_name, (), {})
    return m


@pytest.fixture
def valid_kwargs() -> dict:
    """Build valid kwargs for MorpheInferencer.__init__."""
    # We use monkeypatch in tests that need isinstance to pass.
    # For basic validation tests we pass mocks that will fail isinstance.
    return dict(
        artifact=MagicMock(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        preprocessor=MagicMock(),
        gcnn_inferencer=MagicMock(),
        autoencoder_inferencer=MagicMock(),
        ld_inferencer=MagicMock(),
        pd_inferencer=MagicMock(),
    )


# ── __init__ validation ────────────────────────────────────────────


class TestMorpheInferencerInit:
    def test_rejects_none_artifact(self, valid_kwargs: dict) -> None:
        valid_kwargs["artifact"] = None
        with pytest.raises(ValueError, match="artifact must be a valid MorpheArtifact"):
            MorpheInferencer(**valid_kwargs)

    def test_rejects_none_preprocessor(self, valid_kwargs: dict) -> None:
        valid_kwargs["preprocessor"] = None
        with pytest.raises(ValueError, match="preprocessor must be a valid PreProcessor"):
            MorpheInferencer(**valid_kwargs)

    def test_rejects_none_gcnn_inferencer(self, valid_kwargs: dict) -> None:
        valid_kwargs["gcnn_inferencer"] = None
        with pytest.raises(ValueError, match="gcnn_inferencer must be a valid"):
            MorpheInferencer(**valid_kwargs)

    def test_rejects_none_autoencoder_inferencer(self, valid_kwargs: dict) -> None:
        valid_kwargs["autoencoder_inferencer"] = None
        with pytest.raises(ValueError, match="autoencoder_inferencer must be a valid"):
            MorpheInferencer(**valid_kwargs)

    def test_rejects_none_ld_inferencer(self, valid_kwargs: dict) -> None:
        valid_kwargs["ld_inferencer"] = None
        with pytest.raises(ValueError, match="ld_inferencer must be a valid"):
            MorpheInferencer(**valid_kwargs)

    def test_rejects_none_pd_inferencer(self, valid_kwargs: dict) -> None:
        valid_kwargs["pd_inferencer"] = None
        with pytest.raises(ValueError, match="pd_inferencer must be a valid"):
            MorpheInferencer(**valid_kwargs)


# ── _prepare_regions ───────────────────────────────────────────────


@pytest.fixture
def morphe_inferencer() -> MorpheInferencer:
    """Build a MorpheInferencer bypassing __init__ for unit tests."""
    inf = object.__new__(MorpheInferencer)
    inf.artifact = MagicMock()
    inf.device = torch.device("cpu")
    inf.dtype = torch.float32
    inf.preprocessor = MagicMock()
    inf.gcnn_inferencer = MagicMock()
    inf.autoencoder_inferencer = MagicMock()
    inf.latent_diffusion_inferencer = MagicMock()
    inf.pixel_diffusion_inferencer = MagicMock()
    return inf


class TestPrepareRegions:
    def test_raises_when_root_dir_empty(
        self, morphe_inferencer: MorpheInferencer
    ) -> None:
        import pandas as pd

        with pytest.raises(ValueError, match="Must provide root_dir"):
            morphe_inferencer._prepare_regions(
                df=pd.DataFrame({"a": [1]}), root_dir=""
            )

    def test_calls_preprocessor_and_gcnn(
        self, morphe_inferencer: MorpheInferencer, tmp_path: Path
    ) -> None:
        import pandas as pd

        df = pd.DataFrame({"a": [1]})
        processed_df = pd.DataFrame({"a": [1], "processed": [True]})

        morphe_inferencer.preprocessor.preprocess.return_value = (processed_df, None)
        morphe_inferencer.gcnn_inferencer.predict_proba.return_value = processed_df

        result_df, result_dir = morphe_inferencer._prepare_regions(
            df=df, root_dir=tmp_path / "regions"
        )

        morphe_inferencer.preprocessor.preprocess.assert_called_once()
        morphe_inferencer.gcnn_inferencer.predict_proba.assert_called_once_with(
            df=processed_df
        )
        morphe_inferencer.autoencoder_inferencer.add_rgb_and_rasterize_per_region.assert_called_once()
        assert result_dir == tmp_path / "regions"
        assert result_dir.exists()


# ── _decode_latents_to_cell_maps ───────────────────────────────────


class TestDecodeLatentsToCellMaps:
    def test_decodes_each_latent(self, morphe_inferencer: MorpheInferencer) -> None:
        latent_a = torch.randn(1, 4, 8, 8)
        latent_b = torch.randn(1, 4, 8, 8)

        rgb_a = torch.randn(3, 64, 64)
        rgb_b = torch.randn(3, 64, 64)
        cell_a = torch.randint(0, 5, (1, 64, 64))
        cell_b = torch.randint(0, 5, (1, 64, 64))

        morphe_inferencer.pixel_diffusion_inferencer.decode.side_effect = [rgb_a, rgb_b]
        morphe_inferencer.autoencoder_inferencer.decode_rgb_to_cell_map.side_effect = [
            cell_a,
            cell_b,
        ]

        results = morphe_inferencer._decode_latents_to_cell_maps([latent_a, latent_b])

        assert len(results) == 2
        assert torch.equal(results[0], cell_a)
        assert torch.equal(results[1], cell_b)

    def test_empty_list_returns_empty(self, morphe_inferencer: MorpheInferencer) -> None:
        results = morphe_inferencer._decode_latents_to_cell_maps([])
        assert results == []


# ── run_gapfill type check ─────────────────────────────────────────


class TestRunGapfillTypeCheck:
    def test_raises_when_wrong_inferencer_type(
        self, morphe_inferencer: MorpheInferencer, tmp_path: Path
    ) -> None:
        import pandas as pd

        # Make _prepare_regions a no-op
        morphe_inferencer._prepare_regions = MagicMock()  # type: ignore

        # latent_diffusion_inferencer is a MagicMock, not a GapfillInferencer
        with pytest.raises(TypeError, match="GapfillInferencer"):
            morphe_inferencer.run_gapfill(
                df=pd.DataFrame({"a": [1]}),
                root_dir=tmp_path,
                input_dir=tmp_path,
                save_dir=tmp_path,
            )


class TestRunInpaintTypeCheck:
    def test_raises_when_wrong_inferencer_type(
        self, morphe_inferencer: MorpheInferencer, tmp_path: Path
    ) -> None:
        import pandas as pd

        morphe_inferencer._prepare_regions = MagicMock()  # type: ignore

        with pytest.raises(TypeError, match="InpaintInferencer"):
            morphe_inferencer.run_inpaint(
                df=pd.DataFrame({"a": [1]}),
                root_dir=tmp_path,
                input_dir=tmp_path,
                mask_dir=tmp_path,
            )


class TestRunThreeDimTypeCheck:
    def test_raises_when_wrong_inferencer_type(
        self, morphe_inferencer: MorpheInferencer, tmp_path: Path
    ) -> None:
        import pandas as pd

        morphe_inferencer._prepare_regions = MagicMock()  # type: ignore

        with pytest.raises(TypeError, match="ThreeDimImputationInferencer"):
            morphe_inferencer.run_three_dim_imputation(
                df=pd.DataFrame({"a": [1]}),
                root_dir=tmp_path,
                prev_path=tmp_path / "prev.png",
                next_path=tmp_path / "next.png",
                out_dir=tmp_path / "out",
            )
