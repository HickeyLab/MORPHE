from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from core.pixel_diffusion.artifact import PixelDiffusionArtifact


def _make_artifact(
    *,
    optimizer_state_dict: Mapping | None = None,
) -> PixelDiffusionArtifact:
    return PixelDiffusionArtifact(
        adapter_state_dict={"weight": torch.randn(2, 2)},
        unet_state_dict={"weight": torch.randn(3, 3)},
        adapter_kwargs={"cz": 4, "cond_ch": 64},
        unet_kwargs={"base_ch": 128, "cond_ch": 64},
        scheduler_pretrained="runwayml/stable-diffusion-v1-5",
        scheduler_num_inference_steps=200,
        train_index="train_index.jsonl",
        val_index="val_index.jsonl",
        bs=4,
        lr=1e-5,
        optimizer_betas=(0.9, 0.999),
        optimizer_weight_decay=1e-5,
        optimizer_state_dict=optimizer_state_dict,
        best_epoch=5,
        best_global_step=1000,
    )


def _make_payload(artifact: PixelDiffusionArtifact) -> dict[str, object]:
    return {
        "adapter_state_dict": dict(artifact.adapter_state_dict),
        "unet_state_dict": dict(artifact.unet_state_dict),
        "adapter_kwargs": dict(artifact.adapter_kwargs),
        "unet_kwargs": dict(artifact.unet_kwargs),
        "scheduler_pretrained": artifact.scheduler_pretrained,
        "scheduler_num_inference_steps": artifact.scheduler_num_inference_steps,
        "train_index": artifact.train_index,
        "val_index": artifact.val_index,
        "bs": artifact.bs,
        "lr": artifact.lr,
        "optimizer_betas": artifact.optimizer_betas,
        "optimizer_weight_decay": artifact.optimizer_weight_decay,
        "optimizer_state_dict": artifact.optimizer_state_dict,
        "best_epoch": artifact.best_epoch,
        "best_global_step": artifact.best_global_step,
    }


# ── save / load round-trip ──────────────────────────────────────────


def test_save_and_load_round_trip_preserves_metadata(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "nested" / "artifact.pt"

    artifact.save(save_path)
    assert save_path.exists()

    loaded = PixelDiffusionArtifact.load(save_path)

    assert loaded.scheduler_pretrained == artifact.scheduler_pretrained
    assert loaded.scheduler_num_inference_steps == artifact.scheduler_num_inference_steps
    assert loaded.train_index == artifact.train_index
    assert loaded.val_index == artifact.val_index
    assert loaded.bs == artifact.bs
    assert loaded.lr == artifact.lr
    assert loaded.optimizer_betas == artifact.optimizer_betas
    assert loaded.optimizer_weight_decay == artifact.optimizer_weight_decay
    assert loaded.best_epoch == artifact.best_epoch
    assert loaded.best_global_step == artifact.best_global_step


def test_save_and_load_round_trip_preserves_weights(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "artifact.pt"

    artifact.save(save_path)
    loaded = PixelDiffusionArtifact.load(save_path)

    for key, expected in artifact.adapter_state_dict.items():
        assert torch.equal(loaded.adapter_state_dict[key], expected.cpu())
    for key, expected in artifact.unet_state_dict.items():
        assert torch.equal(loaded.unet_state_dict[key], expected.cpu())


def test_save_and_load_with_optimizer_state(tmp_path: Path) -> None:
    artifact = _make_artifact(optimizer_state_dict={"step": 100, "lr": 1e-5})
    save_path = tmp_path / "artifact.pt"

    artifact.save(save_path)
    loaded = PixelDiffusionArtifact.load(save_path)

    assert loaded.optimizer_state_dict is not None
    assert loaded.optimizer_state_dict["step"] == 100


def test_save_serializes_cpu_tensors(tmp_path: Path) -> None:
    artifact = _make_artifact()
    save_path = tmp_path / "artifact.pt"
    artifact.save(save_path)

    payload = torch.load(save_path, map_location="cpu", weights_only=True)

    for tensor in payload["adapter_state_dict"].values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"

    for tensor in payload["unet_state_dict"].values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"


# ── load validation ─────────────────────────────────────────────────


def test_load_raises_when_required_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.pt"
    path.write_bytes(b"x")

    payload = _make_payload(_make_artifact())
    del payload["scheduler_pretrained"]

    monkeypatch.setattr(torch, "load", lambda *a, **kw: payload)

    with pytest.raises(ValueError, match="missing key 'scheduler_pretrained'"):
        PixelDiffusionArtifact.load(path)


def test_load_falls_back_when_weights_only_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "fallback.pt"
    path.write_bytes(b"x")

    artifact = _make_artifact()
    payload = _make_payload(artifact)
    calls: list[dict[str, object]] = []

    def fake_load(*args, **kwargs):
        calls.append(dict(kwargs))
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword")
        return payload

    monkeypatch.setattr(torch, "load", fake_load)

    loaded = PixelDiffusionArtifact.load(path)

    assert isinstance(loaded, PixelDiffusionArtifact)
    assert len(calls) == 2
    assert "weights_only" in calls[0]
    assert "weights_only" not in calls[1]


# ── _coerce_optimizer_betas ─────────────────────────────────────────


def test_coerce_optimizer_betas_accepts_tuple() -> None:
    result = PixelDiffusionArtifact._coerce_optimizer_betas((0.9, 0.999))
    assert result == (0.9, 0.999)


def test_coerce_optimizer_betas_accepts_list() -> None:
    result = PixelDiffusionArtifact._coerce_optimizer_betas([0.9, 0.999])
    assert result == (0.9, 0.999)


def test_coerce_optimizer_betas_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="sequence of length 2"):
        PixelDiffusionArtifact._coerce_optimizer_betas((0.9,))


def test_coerce_optimizer_betas_rejects_non_sequence() -> None:
    with pytest.raises(ValueError, match="sequence of length 2"):
        PixelDiffusionArtifact._coerce_optimizer_betas(0.9)
