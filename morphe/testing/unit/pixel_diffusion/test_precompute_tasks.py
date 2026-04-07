from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from core.pixel_diffusion.precompute.base import PixelPrecomputeTask
from core.pixel_diffusion.precompute.inpaint import InpaintPrecomputeTask
from core.pixel_diffusion.precompute.outpaint import OutpaintPrecomputeTask
from core.pixel_diffusion.precompute.three_dim import ThreeDimImputationPrecomputeTask


# ═══════════════════════════════════════════════════════════════════
# InpaintPrecomputeTask
# ═══════════════════════════════════════════════════════════════════


class TestInpaintPrecomputeTask:
    def test_is_pixel_precompute_task_subclass(self) -> None:
        assert issubclass(InpaintPrecomputeTask, PixelPrecomputeTask)

    def test_default_fields(self) -> None:
        task = InpaintPrecomputeTask()
        assert task.masks_per_image_train == 2
        assert task.masks_per_image_val == 5
        assert task.img_size == 512
        assert task.ae_pretrained_path == "runwayml/stable-diffusion-v1-5"

    def test_custom_fields(self) -> None:
        task = InpaintPrecomputeTask(
            masks_per_image_train=10,
            masks_per_image_val=3,
            img_size=256,
        )
        assert task.masks_per_image_train == 10
        assert task.masks_per_image_val == 3
        assert task.img_size == 256

    def test_get_encoder_input_returns_masked_img(self) -> None:
        task = InpaintPrecomputeTask()
        masked_img = torch.randn(2, 3, 512, 512)
        img = torch.randn(2, 3, 512, 512)
        mask = torch.randn(2, 1, 512, 512)
        batch = (masked_img, img, mask)

        result = task.get_encoder_input(batch)
        assert torch.equal(result, masked_img)

    def test_get_target_img_returns_full_img(self) -> None:
        task = InpaintPrecomputeTask()
        masked_img = torch.randn(2, 3, 512, 512)
        img = torch.randn(2, 3, 512, 512)
        mask = torch.randn(2, 1, 512, 512)
        batch = (masked_img, img, mask)

        result = task.get_target_img(batch)
        assert torch.equal(result, img)

    def test_get_sample_name_format(self) -> None:
        task = InpaintPrecomputeTask()
        batch = (torch.empty(0), torch.empty(0), torch.empty(0))

        name = task.get_sample_name(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=0,
            global_idx=42,
            split_name="train",
        )
        assert name == "train_0000042"

    def test_get_sample_name_val_split(self) -> None:
        task = InpaintPrecomputeTask()
        batch = (torch.empty(0), torch.empty(0), torch.empty(0))

        name = task.get_sample_name(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=0,
            global_idx=7,
            split_name="val",
        )
        assert name == "val_0000007"

    def test_get_metadata_returns_mask(self) -> None:
        task = InpaintPrecomputeTask()
        mask = torch.randn(4, 1, 512, 512)
        batch = (torch.empty(0), torch.empty(0), mask)

        meta = task.get_metadata(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=2,
            global_idx=0,
            split_name="train",
        )
        assert "mask" in meta
        assert torch.equal(meta["mask"], mask[2].detach().cpu())

    @patch("core.pixel_diffusion.precompute.inpaint.build_inpaint_dataset")
    def test_build_dataset_delegates_to_builder(self, mock_build: MagicMock) -> None:
        train_ds = MagicMock()
        val_ds = MagicMock()
        mock_build.return_value = (train_ds, val_ds)

        task = InpaintPrecomputeTask(
            masks_per_image_train=3,
            masks_per_image_val=7,
            img_size=256,
        )
        result = task.build_dataset(Path("/fake"))

        mock_build.assert_called_once_with(
            root_dir=Path("/fake"),
            masks_per_image_train=3,
            masks_per_image_val=7,
            img_size=256,
        )
        assert result == (train_ds, val_ds)


# ═══════════════════════════════════════════════════════════════════
# OutpaintPrecomputeTask
# ═══════════════════════════════════════════════════════════════════


class TestOutpaintPrecomputeTask:
    def test_is_pixel_precompute_task_subclass(self) -> None:
        assert issubclass(OutpaintPrecomputeTask, PixelPrecomputeTask)

    def test_default_fields(self) -> None:
        task = OutpaintPrecomputeTask()
        assert task.masks_per_image_train == 5
        assert task.masks_per_image_val == 5
        assert task.img_size == 512

    def test_get_encoder_input_returns_masked_img(self) -> None:
        task = OutpaintPrecomputeTask()
        masked_img = torch.randn(2, 3, 512, 512)
        img = torch.randn(2, 3, 512, 512)
        bbox = torch.randn(2, 4)
        batch = (masked_img, img, bbox)

        result = task.get_encoder_input(batch)
        assert torch.equal(result, masked_img)

    def test_get_target_img_returns_full_img(self) -> None:
        task = OutpaintPrecomputeTask()
        masked_img = torch.randn(2, 3, 512, 512)
        img = torch.randn(2, 3, 512, 512)
        bbox = torch.randn(2, 4)
        batch = (masked_img, img, bbox)

        result = task.get_target_img(batch)
        assert torch.equal(result, img)

    def test_get_sample_name_uses_dataset_filenames(self) -> None:
        task = OutpaintPrecomputeTask()
        dataset = MagicMock()
        dataset.masks_per_image = 5
        dataset.img_files = ["/data/region_001.png", "/data/region_002.png"]
        batch = (torch.empty(0), torch.empty(0), torch.empty(0))

        # global_idx=7 → img_idx=1 (7//5), mask_idx=2 (7%5)
        name = task.get_sample_name(
            dataset=dataset,
            batch=batch,
            batch_idx=0,
            global_idx=7,
            split_name="train",
        )
        assert name == "region_002_k002"

    def test_get_sample_name_first_image_first_mask(self) -> None:
        task = OutpaintPrecomputeTask()
        dataset = MagicMock()
        dataset.masks_per_image = 3
        dataset.img_files = ["/data/img_a.png"]
        batch = (torch.empty(0), torch.empty(0), torch.empty(0))

        name = task.get_sample_name(
            dataset=dataset,
            batch=batch,
            batch_idx=0,
            global_idx=0,
            split_name="val",
        )
        assert name == "img_a_k000"

    def test_get_metadata_returns_bbox(self) -> None:
        task = OutpaintPrecomputeTask()
        bbox = torch.tensor([[0.1, 0.2, 0.8, 0.9], [0.3, 0.4, 0.7, 0.6]])
        batch = (torch.empty(0), torch.empty(0), bbox)

        meta = task.get_metadata(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=1,
            global_idx=0,
            split_name="train",
        )
        assert "bbox" in meta
        assert torch.equal(meta["bbox"], bbox[1].detach().cpu())

    @patch("core.pixel_diffusion.precompute.outpaint.build_outpaint_dataset")
    def test_build_dataset_delegates_to_builder(self, mock_build: MagicMock) -> None:
        train_ds = MagicMock()
        val_ds = MagicMock()
        mock_build.return_value = (train_ds, val_ds)

        task = OutpaintPrecomputeTask(
            masks_per_image_train=8,
            masks_per_image_val=4,
            img_size=128,
        )
        result = task.build_dataset(Path("/fake"))

        mock_build.assert_called_once_with(
            root_dir=Path("/fake"),
            masks_per_image_train=8,
            masks_per_image_val=4,
            img_size=128,
        )
        assert result == (train_ds, val_ds)


# ═══════════════════════════════════════════════════════════════════
# ThreeDimImputationPrecomputeTask
# ═══════════════════════════════════════════════════════════════════


class TestThreeDimImputationPrecomputeTask:
    def test_is_pixel_precompute_task_subclass(self) -> None:
        assert issubclass(ThreeDimImputationPrecomputeTask, PixelPrecomputeTask)

    def test_get_encoder_input_returns_prev_image(self) -> None:
        task = ThreeDimImputationPrecomputeTask()
        img_prev = torch.randn(2, 3, 512, 512)
        img_next = torch.randn(2, 3, 512, 512)
        img_gt = torch.randn(2, 3, 512, 512)
        w_prev = torch.tensor([0.3, 0.5])
        w_next = torch.tensor([0.7, 0.5])
        batch = (img_prev, img_next, img_gt, w_prev, w_next)

        result = task.get_encoder_input(batch)
        assert torch.equal(result, img_prev)

    def test_get_target_img_returns_gt_image(self) -> None:
        task = ThreeDimImputationPrecomputeTask()
        img_prev = torch.randn(2, 3, 512, 512)
        img_next = torch.randn(2, 3, 512, 512)
        img_gt = torch.randn(2, 3, 512, 512)
        w_prev = torch.tensor([0.3, 0.5])
        w_next = torch.tensor([0.7, 0.5])
        batch = (img_prev, img_next, img_gt, w_prev, w_next)

        result = task.get_target_img(batch)
        assert torch.equal(result, img_gt)

    def test_get_sample_name_format(self) -> None:
        task = ThreeDimImputationPrecomputeTask()
        batch = (torch.empty(0),) * 5

        name = task.get_sample_name(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=0,
            global_idx=123,
            split_name="train",
        )
        assert name == "train_0000123"

    def test_get_sample_name_val_split(self) -> None:
        task = ThreeDimImputationPrecomputeTask()
        batch = (torch.empty(0),) * 5

        name = task.get_sample_name(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=0,
            global_idx=5,
            split_name="val",
        )
        assert name == "val_0000005"

    def test_get_metadata_returns_weights(self) -> None:
        task = ThreeDimImputationPrecomputeTask()
        w_prev = torch.tensor([0.3, 0.7])
        w_next = torch.tensor([0.7, 0.3])
        batch = (torch.empty(0), torch.empty(0), torch.empty(0), w_prev, w_next)

        meta = task.get_metadata(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=1,
            global_idx=0,
            split_name="train",
        )
        assert "w_prev" in meta
        assert "w_next" in meta
        assert meta["w_prev"] == pytest.approx(0.7)
        assert meta["w_next"] == pytest.approx(0.3)

    def test_get_metadata_returns_floats(self) -> None:
        task = ThreeDimImputationPrecomputeTask()
        w_prev = torch.tensor([0.5])
        w_next = torch.tensor([0.5])
        batch = (torch.empty(0), torch.empty(0), torch.empty(0), w_prev, w_next)

        meta = task.get_metadata(
            dataset=MagicMock(),
            batch=batch,
            batch_idx=0,
            global_idx=0,
            split_name="val",
        )
        assert isinstance(meta["w_prev"], float)
        assert isinstance(meta["w_next"], float)

    @patch("core.pixel_diffusion.precompute.three_dim.build_three_dim_dataset")
    def test_build_dataset_delegates_to_builder(self, mock_build: MagicMock) -> None:
        train_ds = MagicMock()
        val_ds = MagicMock()
        mock_build.return_value = (train_ds, val_ds)

        task = ThreeDimImputationPrecomputeTask()
        result = task.build_dataset(Path("/fake"))

        mock_build.assert_called_once_with(root_dir=Path("/fake"))
        assert result == (train_ds, val_ds)
