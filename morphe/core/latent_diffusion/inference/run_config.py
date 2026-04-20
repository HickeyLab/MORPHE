from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path
from typing import Literal

import torch


DIRECTION = Literal["left", "right", "up", "down"]


@dataclass(slots=True)
class LatentBaseRunConfig:
    def validate(self) -> None:
        raise NotImplementedError

@dataclass(slots=True)
class InpaintRunConfig(LatentBaseRunConfig):
    input_dir: str | Path
    mask_dir: str | Path
    num_steps: int = 200
    show_plot: bool = False
    plot_title: str = "Inpainting Progress"
    plot_fig_size: tuple[int, int] = (6,6)

    def validate(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.mask_dir = Path(self.mask_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"mask_dir does not exist: {self.mask_dir}")

        if not isinstance(self.num_steps, int):
            raise TypeError("num_steps must be an int.")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        if not isinstance(self.show_plot, bool):
            raise TypeError("show_plot must be a bool.")

        if self.plot_title is not None and not isinstance(self.plot_title, str):
            raise TypeError("plot_title must be a str or None.")

        if self.plot_fig_size is not None:
            if (
                not isinstance(self.plot_fig_size, tuple)
                or len(self.plot_fig_size) != 2
                or not all(isinstance(x, int) for x in self.plot_fig_size)
            ):
                raise TypeError("plot_fig_size must be a tuple[int, int] or None.")
            if any(x <= 0 for x in self.plot_fig_size):
                raise ValueError("plot_fig_size values must be > 0.")


@dataclass(slots=True)
class GapfillRunConfig(LatentBaseRunConfig):
    input_dir: str | Path
    save_dir: str | Path
    save_name: str = "default"
    num_steps: int = 200
    iterations: int = 10
    show_plot: bool = False
    plot_title: str | None = None
    plot_fig_size: tuple[int, int] = (6,6)
    bbox: torch.Tensor | None = None
    
    def __post_init__(self):
        if self.plot_title is None:
            self.plot_title = f"Gapfill ({self.iterations} iterations, {self.num_steps} num_steps)"
            
        self.input_dir = Path(self.input_dir)
        self.save_dir = Path(self.save_dir)
        
        self.validate()

    def validate(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.save_dir = Path(self.save_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")

        if not isinstance(self.save_name, str):
            raise TypeError("save_name must be a str.")
        if not self.save_name.strip():
            raise ValueError("save_name must be non-empty.")

        if not isinstance(self.num_steps, int):
            raise TypeError("num_steps must be an int.")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        if not isinstance(self.iterations, int):
            raise TypeError("iterations must be an int.")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1.")

        if not isinstance(self.show_plot, bool):
            raise TypeError("show_plot must be a bool.")

        if self.plot_title is not None and not isinstance(self.plot_title, str):
            raise TypeError("plot_title must be a str or None.")

        if self.plot_fig_size is not None:
            if (
                not isinstance(self.plot_fig_size, tuple)
                or len(self.plot_fig_size) != 2
                or not all(isinstance(x, int) for x in self.plot_fig_size)
            ):
                raise TypeError("plot_fig_size must be a tuple[int, int] or None.")
            if any(x <= 0 for x in self.plot_fig_size):
                raise ValueError("plot_fig_size values must be > 0.")

        if self.bbox is not None:
            if not isinstance(self.bbox, torch.Tensor):
                raise TypeError("bbox must be a torch.Tensor or None.")

        self.save_dir.mkdir(parents=True, exist_ok=True)

@dataclass(slots=True)
class OutpaintRunConfig(LatentBaseRunConfig):
    input_dir: str | Path
    save_dir: str | Path
    save_name: str = "default"
    num_steps: int = 200
    crop_ratio: float = 0.97
    iterations: int = 10
    direction: str = "right"
    show_plot: bool = False
    plot_title: str | None = None
    plot_fig_size: tuple[int, int] = (6,6)
    
    def __post_init__(self):
        if self.plot_title is None:
            self.plot_title = f"Outpaint ({self.iterations} iterations, {self.num_steps} num_steps)"

    def validate(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.save_dir = Path(self.save_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")

        if not isinstance(self.save_name, str):
            raise TypeError("save_name must be a str.")
        if not self.save_name.strip():
            raise ValueError("save_name must be non-empty.")

        if not isinstance(self.num_steps, int):
            raise TypeError("num_steps must be an int.")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        if not isinstance(self.crop_ratio, (int, float)):
            raise TypeError("crop_ratio must be a float.")
        if not (0.0 < float(self.crop_ratio) <= 1.0):
            raise ValueError("crop_ratio must be in (0, 1].")

        if not isinstance(self.iterations, int):
            raise TypeError("iterations must be an int.")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1.")

        if self.direction not in {"left", "right", "up", "down"}:
            raise ValueError("direction must be one of: 'left', 'right', 'up', 'down'.")

        if not isinstance(self.show_plot, bool):
            raise TypeError("show_plot must be a bool.")

        if self.plot_title is not None and not isinstance(self.plot_title, str):
            raise TypeError("plot_title must be a str or None.")

        if self.plot_fig_size is not None:
            if (
                not isinstance(self.plot_fig_size, tuple)
                or len(self.plot_fig_size) != 2
                or not all(isinstance(x, int) for x in self.plot_fig_size)
            ):
                raise TypeError("plot_fig_size must be a tuple[int, int] or None.")
            if any(x <= 0 for x in self.plot_fig_size):
                raise ValueError("plot_fig_size values must be > 0.")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        
@dataclass(slots=True)
class ThreeDimImputationRunConfig(LatentBaseRunConfig):
    prev_img_path: str | Path
    next_img_path: str | Path
    save_dir: str | Path
    num_inference_steps: int = 200
    w_prev: float = 0.5
    w_next: float = 0.5
    save_latents_name: str = "latents.pt"
    save_png_name: str = "output.png"

    def validate(self) -> None:
        self.prev_img_path = Path(self.prev_img_path)
        self.next_img_path = Path(self.next_img_path)
        self.save_dir = Path(self.save_dir)
        
        if not self.prev_img_path.exists():
            raise FileNotFoundError(f"prev_img_path does not exist: {self.prev_img_path}")
        if not self.next_img_path.exists():
            raise FileNotFoundError(f"next_img_path does not exist: {self.next_img_path}")

        if not isinstance(self.num_inference_steps, int):
            raise TypeError("num_inference_steps must be an int.")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")

        if not isinstance(self.w_prev, (int, float)):
            raise TypeError("w_prev must be a float.")
        if not isinstance(self.w_next, (int, float)):
            raise TypeError("w_next must be a float.")
        if float(self.w_prev) < 0 or float(self.w_prev) > 1:
            raise ValueError("w_prev must be between 0 and 1.")
        if float(self.w_next) < 0 or float(self.w_next) > 1:
            raise ValueError("w_next must be between 0 and 1.")

        if not isinstance(self.save_latents_name, str):
            raise TypeError("save_latents_name must be a str.")
        if not self.save_latents_name.strip():
            raise ValueError("save_latents_name must be non-empty.")

        if not isinstance(self.save_png_name, str):
            raise TypeError("save_png_name must be a str.")
        if not self.save_png_name.strip():
            raise ValueError("save_png_name must be non-empty.")

        self.save_dir.mkdir(parents=True, exist_ok=True)
        
@dataclass(slots=True)
class ThreeDimImputationWeightSweepConfig(LatentBaseRunConfig):
    prev_img_path: Path | str
    next_img_path: Path | str
    save_dir: Path | str = Path("./outputs")
    num_inference_steps: int = 200
    start: float = 0.1
    end: float = 0.9
    step: float = 0.1

    def validate(self) -> None:
        self.prev_img_path = Path(self.prev_img_path)
        self.next_img_path = Path(self.next_img_path)
        self.save_dir = Path(self.save_dir)

        if not self.prev_img_path.exists():
            raise FileNotFoundError(f"prev_img_path does not exist: {self.prev_img_path}")
        if not self.prev_img_path.is_file():
            raise ValueError(f"prev_img_path must be a file: {self.prev_img_path}")

        if not self.next_img_path.exists():
            raise FileNotFoundError(f"next_img_path does not exist: {self.next_img_path}")
        if not self.next_img_path.is_file():
            raise ValueError(f"next_img_path must be a file: {self.next_img_path}")

        if not isinstance(self.num_inference_steps, int):
            raise TypeError("num_inference_steps must be an int.")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be > 0.")

        for name, value in (
            ("start", self.start),
            ("end", self.end),
            ("step", self.step),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float.")
            if name != "step" and not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0].")

        if not (0.0 < float(self.step) <= 1.0):
            raise ValueError("step must be in (0.0, 1.0].")

        if float(self.start) > float(self.end):
            raise ValueError("start must be <= end.")

        span = float(self.end) - float(self.start)
        if self.step > span and self.start != self.end:
            raise ValueError("step is larger than the sweep range.")
        
        self.save_dir.mkdir(parents=True, exist_ok=True)