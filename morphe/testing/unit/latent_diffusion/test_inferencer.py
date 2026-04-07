from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from core.latent_diffusion.inference.gapfill import GapfillInferencer
from core.latent_diffusion.inference.inpaint import InpaintInferencer
from core.latent_diffusion.inference.three_dim import ThreeDimImputationInferencer
from core.latent_diffusion.inference.base import _get_scaling_factor


# ── helpers ─────────────────────────────────────────────────────────


class _FakeLatentDist:
    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def sample(self) -> torch.Tensor:
        return self._value


class _FakeEncodeOutput:
    def __init__(self, value: torch.Tensor) -> None:
        self.latent_dist = _FakeLatentDist(value)


class _FakeDecodeOutput:
    def __init__(self, value: torch.Tensor) -> None:
        self.sample = value


def _make_dummy_vae(latent_h: int = 8, latent_w: int = 8) -> MagicMock:
    vae = MagicMock()
    vae.config = MagicMock(scaling_factor=0.18215)
    vae.eval = MagicMock(return_value=vae)

    def _encode(x: torch.Tensor) -> _FakeEncodeOutput:
        b = x.shape[0]
        return _FakeEncodeOutput(torch.randn(b, 4, latent_h, latent_w))

    def _decode(x: torch.Tensor) -> _FakeDecodeOutput:
        b = x.shape[0]
        return _FakeDecodeOutput(torch.randn(b, 3, latent_h * 8, latent_w * 8))

    vae.encode = _encode
    vae.decode = _decode
    return vae


def _make_dummy_unet() -> MagicMock:
    unet = MagicMock()
    unet.eval = MagicMock(return_value=unet)

    def _forward(x: torch.Tensor, t, encoder_hidden_states=None) -> MagicMock:
        out = MagicMock()
        out.sample = torch.randn_like(x)
        return out

    unet.__call__ = _forward
    unet.side_effect = _forward
    return unet


def _make_dummy_cond_encoder(out_dim: int = 32) -> MagicMock:
    enc = MagicMock(spec=torch.nn.Module)
    enc.eval = MagicMock(return_value=enc)

    def _forward(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.randn(b, 16, out_dim)

    enc.__call__ = _forward
    enc.side_effect = _forward
    return enc


def _make_dummy_bbox_encoder(out_dim: int = 16) -> MagicMock:
    enc = MagicMock()
    enc.eval = MagicMock(return_value=enc)

    def _forward(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.randn(b, out_dim)

    enc.__call__ = _forward
    enc.side_effect = _forward
    return enc


def _make_dummy_coord_encoder(out_dim: int = 16) -> MagicMock:
    enc = MagicMock()
    enc.eval = MagicMock(return_value=enc)

    def _forward(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.randn(b, 16, out_dim)

    enc.__call__ = _forward
    enc.side_effect = _forward
    return enc


def _make_dummy_scheduler() -> MagicMock:
    sched = MagicMock()
    sched.set_timesteps = MagicMock()
    sched.timesteps = [torch.tensor(999), torch.tensor(500), torch.tensor(0)]

    def _add_noise(sample, noise, timestep):
        return sample + noise * 0.01

    def _step(model_output, timestep, sample):
        out = MagicMock()
        out.prev_sample = sample - model_output * 0.01
        return out

    sched.add_noise = _add_noise
    sched.step = _step
    return sched


@dataclass
class _FakeArtifact:
    inference_mode: str
    img_size: int | None = 64

    class architecture:
        @staticmethod
        def build_components(device, dtype):
            pass


# ── _get_scaling_factor ─────────────────────────────────────────────


def test_get_scaling_factor_from_config() -> None:
    vae = _make_dummy_vae()
    assert _get_scaling_factor(vae) == pytest.approx(0.18215)


def test_get_scaling_factor_fallback_when_no_config() -> None:
    vae = MagicMock(spec=[])
    assert _get_scaling_factor(vae) == pytest.approx(0.18215)


# ═══════════════════════════════════════════════════════════════════
# GapfillInferencer
# ═══════════════════════════════════════════════════════════════════


class TestGapfillInit:
    def test_wrong_inference_mode_rejected(self) -> None:
        artifact = _FakeArtifact(inference_mode="inpainting")
        with pytest.raises(RuntimeError, match="GapfillInferencer requires"):
            GapfillInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=_make_dummy_cond_encoder(),
                coord_encoder=None,
                bbox_encoder=_make_dummy_bbox_encoder(),
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_missing_bbox_encoder_rejected(self) -> None:
        artifact = _FakeArtifact(inference_mode="gapfill")
        with pytest.raises(RuntimeError, match="bbox encoder"):
            GapfillInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=_make_dummy_cond_encoder(),
                coord_encoder=None,
                bbox_encoder=None,  # type: ignore
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_missing_img_size_rejected(self) -> None:
        artifact = _FakeArtifact(inference_mode="gapfill", img_size=None)
        with pytest.raises(RuntimeError, match="img_size"):
            GapfillInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=_make_dummy_cond_encoder(),
                coord_encoder=None,
                bbox_encoder=_make_dummy_bbox_encoder(),
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


@pytest.fixture
def gapfill_inferencer() -> GapfillInferencer:
    """Build a GapfillInferencer bypassing __init__ validation for unit tests."""
    artifact = _FakeArtifact(inference_mode="gapfill", img_size=64)

    inf = object.__new__(GapfillInferencer)
    inf.artifact = artifact  # type: ignore
    inf.unet = _make_dummy_unet()
    inf.vae = _make_dummy_vae()
    inf.cond_encoder = _make_dummy_cond_encoder()
    inf.coord_encoder = None
    inf.bbox_encoder = _make_dummy_bbox_encoder()
    inf.noise_scheduler = _make_dummy_scheduler()
    inf.device = torch.device("cpu")
    inf.dtype = torch.float32
    inf.scaling_factor = 0.18215
    return inf


class TestGapfillCreateLatentMask:
    def test_mask_shape(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        mask = gapfill_inferencer._create_latent_mask(
            bbox, latent_shape=(1, 4, 8, 8)
        )
        assert mask.shape == (1, 1, 8, 8)

    def test_full_bbox_all_ones(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        mask = gapfill_inferencer._create_latent_mask(
            bbox, latent_shape=(1, 4, 8, 8)
        )
        # With a full [0,0,1,1] bbox, most pixels should be masked
        assert mask.sum() > 0

    def test_zero_bbox_all_zeros(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        mask = gapfill_inferencer._create_latent_mask(
            bbox, latent_shape=(1, 4, 8, 8)
        )
        # Point bbox should produce zero or near-zero mask
        assert mask.sum() <= 1

    def test_multiple_bboxes(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
        mask = gapfill_inferencer._create_latent_mask(
            bbox, latent_shape=(1, 4, 8, 8)
        )
        assert mask.shape == (2, 1, 8, 8)


class TestGapfillResolveBbox:
    def test_returns_provided_bbox(self, gapfill_inferencer: GapfillInferencer) -> None:
        from core.latent_diffusion.inference.run_config import GapfillRunConfig

        custom_bbox = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        cfg = GapfillRunConfig(
            input_dir="/tmp/fake",
            save_dir="/tmp/fake",
            bbox=custom_bbox,
        )
        result = gapfill_inferencer._resolve_bbox(cfg)
        assert torch.equal(result, custom_bbox)

    def test_returns_default_when_none(self, gapfill_inferencer: GapfillInferencer) -> None:
        from core.latent_diffusion.inference.run_config import GapfillRunConfig

        cfg = GapfillRunConfig(
            input_dir="/tmp/fake",
            save_dir="/tmp/fake",
            bbox=None,
        )
        result = gapfill_inferencer._resolve_bbox(cfg)
        assert result.shape == (1, 4)
        assert result[0, 1].item() == pytest.approx(0.4375)


class TestGapfillGetGapColumns:
    def test_full_width_bbox(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        start, end = gapfill_inferencer._get_gap_columns(bbox, latent_width=8)
        assert start == 0
        assert end == 8

    def test_half_width_bbox(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([[0.25, 0.0, 0.75, 1.0]])
        start, end = gapfill_inferencer._get_gap_columns(bbox, latent_width=8)
        assert start == 2
        assert end == 6

    def test_wrong_bbox_shape_rejected(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([0.0, 0.0, 1.0, 1.0])  # 1D, not 2D
        with pytest.raises(ValueError, match="Expected bbox with shape"):
            gapfill_inferencer._get_gap_columns(bbox, latent_width=8)

    def test_empty_interval_rejected(self, gapfill_inferencer: GapfillInferencer) -> None:
        bbox = torch.tensor([[0.5, 0.0, 0.5, 1.0]])  # x1 == x2
        with pytest.raises(ValueError, match="invalid latent gap interval"):
            gapfill_inferencer._get_gap_columns(bbox, latent_width=8)


class TestGapfillSaveFinalOutputs:
    def test_saves_latent_and_image(
        self, gapfill_inferencer: GapfillInferencer, tmp_path: Path
    ) -> None:
        latent = torch.randn(1, 4, 8, 8)
        preview = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        gapfill_inferencer._save_final_outputs(
            latent=latent,
            preview=preview,
            iteration=0,
            save_dir=tmp_path,
            save_name="test_run",
        )

        run_dir = tmp_path / "test_run"
        assert (run_dir / "0_latent.pt").exists()
        assert (run_dir / "0_image.png").exists()

    def test_noop_when_save_dir_empty(
        self, gapfill_inferencer: GapfillInferencer
    ) -> None:
        # Should not raise
        gapfill_inferencer._save_final_outputs(
            latent=torch.randn(1, 4, 8, 8),
            preview=np.zeros((8, 8, 3), dtype=np.uint8),
            iteration=0,
            save_dir="",
            save_name="test",
        )


class TestGapfillStitchNextLatent:
    def test_output_preserves_width(self, gapfill_inferencer: GapfillInferencer) -> None:
        latent = torch.randn(1, 4, 8, 16)
        latent_input = torch.randn(1, 4, 8, 16)
        bbox = torch.tensor([[0.25, 0.0, 0.75, 1.0]])

        result = gapfill_inferencer._stitch_next_latent(
            current_latent=latent,
            latent_input=latent_input,
            bbox=bbox,
        )
        assert result.shape == latent.shape

    def test_narrow_gap_rejected(self, gapfill_inferencer: GapfillInferencer) -> None:
        latent = torch.randn(1, 4, 8, 8)
        latent_input = torch.randn(1, 4, 8, 8)
        # Very narrow bbox → gap_width < 2
        bbox = torch.tensor([[0.0, 0.0, 0.125, 1.0]])  # 1 pixel wide

        with pytest.raises(ValueError, match="[Gg]ap width"):
            gapfill_inferencer._stitch_next_latent(
                current_latent=latent,
                latent_input=latent_input,
                bbox=bbox,
            )


# ═══════════════════════════════════════════════════════════════════
# InpaintInferencer
# ═══════════════════════════════════════════════════════════════════


class TestInpaintInit:
    def test_wrong_inference_mode_rejected(self) -> None:
        artifact = _FakeArtifact(inference_mode="gapfill")
        with pytest.raises(RuntimeError, match="InpaintInferencer requires"):
            InpaintInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=_make_dummy_cond_encoder(),
                coord_encoder=_make_dummy_coord_encoder(),
                bbox_encoder=None,
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_missing_coord_encoder_rejected(self) -> None:
        artifact = _FakeArtifact(inference_mode="inpainting")
        with pytest.raises(RuntimeError, match="coord_encoder"):
            InpaintInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=_make_dummy_cond_encoder(),
                coord_encoder=None,  # type: ignore
                bbox_encoder=None,
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_missing_img_size_rejected(self) -> None:
        artifact = _FakeArtifact(inference_mode="inpainting", img_size=None)
        with pytest.raises(RuntimeError, match="img_size"):
            InpaintInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=_make_dummy_cond_encoder(),
                coord_encoder=_make_dummy_coord_encoder(),
                bbox_encoder=None,
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


@pytest.fixture
def inpaint_inferencer() -> InpaintInferencer:
    """Build an InpaintInferencer bypassing __init__ validation."""
    artifact = _FakeArtifact(inference_mode="inpainting", img_size=64)

    inf = object.__new__(InpaintInferencer)
    inf.artifact = artifact  # type: ignore
    inf.unet = _make_dummy_unet()
    inf.vae = _make_dummy_vae()
    inf.cond_encoder = _make_dummy_cond_encoder()
    inf.coord_encoder = _make_dummy_coord_encoder()
    inf.bbox_encoder = None
    inf.noise_scheduler = _make_dummy_scheduler()
    inf.device = torch.device("cpu")
    inf.dtype = torch.float32
    inf.scaling_factor = 0.18215
    return inf


class TestInpaintLoadImage:
    def test_load_image_shape(
        self, inpaint_inferencer: InpaintInferencer, tmp_path: Path
    ) -> None:
        from PIL import Image

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(path)

        result = inpaint_inferencer._load_image(path)
        assert result.shape == (1, 3, 64, 64)
        assert result.device.type == "cpu"

    def test_load_image_normalized_range(
        self, inpaint_inferencer: InpaintInferencer, tmp_path: Path
    ) -> None:
        from PIL import Image

        img = Image.fromarray(
            np.full((64, 64, 3), 128, dtype=np.uint8)
        )
        path = tmp_path / "mid.png"
        img.save(path)

        result = inpaint_inferencer._load_image(path)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5


class TestInpaintLoadMask:
    def test_load_mask_shape(
        self, inpaint_inferencer: InpaintInferencer, tmp_path: Path
    ) -> None:
        from PIL import Image

        mask = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        path = tmp_path / "mask.png"
        mask.save(path)

        result = inpaint_inferencer._load_mask(path)
        assert result.shape == (1, 1, 64, 64)

    def test_white_mask_is_ones(
        self, inpaint_inferencer: InpaintInferencer, tmp_path: Path
    ) -> None:
        from PIL import Image

        mask = Image.fromarray(np.full((64, 64), 255, dtype=np.uint8))
        path = tmp_path / "white.png"
        mask.save(path)

        result = inpaint_inferencer._load_mask(path)
        assert result.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_black_mask_is_zeros(
        self, inpaint_inferencer: InpaintInferencer, tmp_path: Path
    ) -> None:
        from PIL import Image

        mask = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        path = tmp_path / "black.png"
        mask.save(path)

        result = inpaint_inferencer._load_mask(path)
        assert result.mean().item() == pytest.approx(0.0, abs=1e-5)


class TestInpaintRequireCoordEncoder:
    def test_returns_encoder_when_present(
        self, inpaint_inferencer: InpaintInferencer
    ) -> None:
        result = inpaint_inferencer._require_coord_encoder()
        assert result is inpaint_inferencer.coord_encoder

    def test_raises_when_none(self, inpaint_inferencer: InpaintInferencer) -> None:
        inpaint_inferencer.coord_encoder = None
        with pytest.raises(RuntimeError, match="coord_encoder is required"):
            inpaint_inferencer._require_coord_encoder()


class TestInpaintToNumpyImage:
    def test_4d_input(self, inpaint_inferencer: InpaintInferencer) -> None:
        t = torch.randn(1, 3, 8, 8)
        result = inpaint_inferencer._to_numpy_image(t)
        assert result.shape == (8, 8, 3)

    def test_3d_input(self, inpaint_inferencer: InpaintInferencer) -> None:
        t = torch.randn(3, 8, 8)
        result = inpaint_inferencer._to_numpy_image(t)
        assert result.shape == (8, 8, 3)

    def test_single_channel_3d(self, inpaint_inferencer: InpaintInferencer) -> None:
        t = torch.randn(1, 8, 8)
        result = inpaint_inferencer._to_numpy_image(t)
        assert result.shape == (8, 8, 1)


class TestInpaintRunDirectoryValidation:
    def test_run_raises_for_nonexistent_input_dir(
        self, inpaint_inferencer: InpaintInferencer, tmp_path: Path
    ) -> None:
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="input_dir"):
            inpaint_inferencer.run(
                input_dir=tmp_path / "nonexistent",
                mask_dir=mask_dir,
            )

    def test_run_raises_for_nonexistent_mask_dir(
        self, inpaint_inferencer: InpaintInferencer, tmp_path: Path
    ) -> None:
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="mask_dir"):
            inpaint_inferencer.run(
                input_dir=input_dir,
                mask_dir=tmp_path / "nonexistent",
            )


# ═══════════════════════════════════════════════════════════════════
# ThreeDimImputationInferencer
# ═══════════════════════════════════════════════════════════════════


class TestThreeDimInit:
    def test_wrong_inference_mode_rejected(self) -> None:
        artifact = _FakeArtifact(inference_mode="gapfill")
        with pytest.raises(RuntimeError, match="ThreeDimImputationInferencer requires"):
            ThreeDimImputationInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=MagicMock(spec=torch.nn.Module),
                coord_encoder=None,
                bbox_encoder=None,
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_missing_cond_encoder_rejected(self) -> None:
        artifact = _FakeArtifact(
            inference_mode="three_dimensional_imputation"
        )
        with pytest.raises(RuntimeError, match="cond_encoder"):
            ThreeDimImputationInferencer(
                artifact=artifact,  # type: ignore
                unet=_make_dummy_unet(),
                vae=_make_dummy_vae(),
                cond_encoder=None,  # type: ignore
                coord_encoder=None,
                bbox_encoder=None,
                noise_scheduler=_make_dummy_scheduler(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


@pytest.fixture
def threedim_inferencer() -> ThreeDimImputationInferencer:
    """Build a ThreeDimImputationInferencer bypassing __init__ validation."""
    artifact = _FakeArtifact(
        inference_mode="three_dimensional_imputation", img_size=64
    )

    inf = object.__new__(ThreeDimImputationInferencer)
    inf.artifact = artifact  # type: ignore
    inf.unet = _make_dummy_unet()
    inf.vae = _make_dummy_vae()
    inf.cond_encoder = _make_dummy_cond_encoder()
    inf.coord_encoder = None
    inf.bbox_encoder = None
    inf.noise_scheduler = _make_dummy_scheduler()
    inf.device = torch.device("cpu")
    inf.dtype = torch.float32
    inf.scaling_factor = 0.18215
    inf.transform = MagicMock(return_value=torch.randn(3, 64, 64))
    return inf


class TestThreeDimDenormalize:
    def test_maps_minus_one_to_zero(
        self, threedim_inferencer: ThreeDimImputationInferencer
    ) -> None:
        t = torch.tensor([-1.0])
        result = threedim_inferencer._denormalize(t)
        assert result.item() == pytest.approx(0.0)

    def test_maps_one_to_one(
        self, threedim_inferencer: ThreeDimImputationInferencer
    ) -> None:
        t = torch.tensor([1.0])
        result = threedim_inferencer._denormalize(t)
        assert result.item() == pytest.approx(1.0)

    def test_maps_zero_to_half(
        self, threedim_inferencer: ThreeDimImputationInferencer
    ) -> None:
        t = torch.tensor([0.0])
        result = threedim_inferencer._denormalize(t)
        assert result.item() == pytest.approx(0.5)

    def test_clamps_out_of_range(
        self, threedim_inferencer: ThreeDimImputationInferencer
    ) -> None:
        t = torch.tensor([-2.0, 2.0])
        result = threedim_inferencer._denormalize(t)
        assert result[0].item() == pytest.approx(0.0)
        assert result[1].item() == pytest.approx(1.0)


class TestThreeDimLoadImageTensor:
    def test_shape_and_device(
        self, threedim_inferencer: ThreeDimImputationInferencer, tmp_path: Path
    ) -> None:
        from PIL import Image

        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        path = tmp_path / "test.png"
        img.save(path)

        result = threedim_inferencer._load_image_tensor(path)
        assert result.shape == (1, 3, 64, 64)
        assert result.device.type == "cpu"


class TestThreeDimSaveOutputs:
    def test_creates_latent_and_png(
        self, threedim_inferencer: ThreeDimImputationInferencer, tmp_path: Path
    ) -> None:
        latents = torch.randn(1, 4, 8, 8)
        decoded = torch.randn(1, 3, 64, 64)

        threedim_inferencer._save_outputs(
            latents=latents,
            decoded=decoded,
            out_dir=tmp_path / "out",
            save_latents_name="latents.pt",
            save_png_name="output.png",
        )

        out_dir = tmp_path / "out"
        assert (out_dir / "latents.pt").exists()
        assert (out_dir / "output.png").exists()

        loaded = torch.load(out_dir / "latents.pt", weights_only=True)
        assert loaded.shape == latents.shape

    def test_creates_parent_dirs(
        self, threedim_inferencer: ThreeDimImputationInferencer, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "nested" / "deep" / "dir"
        threedim_inferencer._save_outputs(
            latents=torch.randn(1, 4, 8, 8),
            decoded=torch.randn(1, 3, 64, 64),
            out_dir=out_dir,
            save_latents_name="l.pt",
            save_png_name="o.png",
        )
        assert out_dir.exists()
