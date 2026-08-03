from __future__ import annotations

import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _nearest_empty_pixel(
    x: float,
    y: float,
    occupied: np.ndarray,
) -> tuple[int, int] | None:
    """Return the empty pixel nearest to a projected floating-point position.

    Pixels are ordered by squared Euclidean distance to ``(x, y)``. Equal
    distances are resolved deterministically by row and then column. The search
    examines successively larger square rings and stops once every unexamined
    pixel is guaranteed to be farther away.
    """
    height, width = occupied.shape
    start_x = int(round(x))
    start_y = int(round(y))
    if not (0 <= start_x < width and 0 <= start_y < height):
        return None

    best: tuple[float, int, int] | None = None
    max_radius = max(start_x, width - 1 - start_x, start_y, height - 1 - start_y)

    def consider(pixel_x: int, pixel_y: int) -> None:
        nonlocal best
        if occupied[pixel_y, pixel_x]:
            return
        candidate = (
            (pixel_x - x) ** 2 + (pixel_y - y) ** 2,
            pixel_y,
            pixel_x,
        )
        if best is None or candidate < best:
            best = candidate

    for radius in range(max_radius + 1):
        left = start_x - radius
        right = start_x + radius
        top = start_y - radius
        bottom = start_y + radius

        for pixel_x in {left, right}:
            if 0 <= pixel_x < width:
                for pixel_y in range(max(0, top), min(height - 1, bottom) + 1):
                    consider(pixel_x, pixel_y)

        # Exclude the corners because the vertical edges above already tested
        # them. A set also prevents duplicate work when top == bottom.
        for pixel_y in {top, bottom}:
            if 0 <= pixel_y < height:
                for pixel_x in range(
                    max(0, left + 1), min(width - 1, right - 1) + 1
                ):
                    consider(pixel_x, pixel_y)

        if best is None:
            continue

        outside_distances = []
        if left - 1 >= 0:
            outside_distances.append((left - 1 - x) ** 2 + (start_y - y) ** 2)
        if right + 1 < width:
            outside_distances.append((right + 1 - x) ** 2 + (start_y - y) ** 2)
        if top - 1 >= 0:
            outside_distances.append((top - 1 - y) ** 2 + (start_x - x) ** 2)
        if bottom + 1 < height:
            outside_distances.append((bottom + 1 - y) ** 2 + (start_x - x) ** 2)

        if not outside_distances or best[0] < min(outside_distances):
            return best[2], best[1]

    return None if best is None else (best[2], best[1])


def rasterize_rgb_regions(
    result_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    split: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
    region_col: str = "unique_region",
    x_col: str = "x",
    y_col: str = "y",
    r_col: str = "R",
    g_col: str = "G",
    b_col: str = "B",
    image_size: int = 512,
    filename_prefix: str = "region",
    nearest_neighbor: bool = False,
) -> None:
    """Rasterize per-region RGB data into image files on disk.

    When ``split=True`` (training), regions are split into train/val
    subdirectories. When ``split=False`` (inference), all regions are written
    flat into ``save_dir``.

    By default, this retains the original collision behavior: a later cell at
    the same rounded pixel overwrites the earlier cell. With
    ``nearest_neighbor=True``, the first cell keeps the rounded pixel and each
    later colliding cell is assigned to the nearest currently empty pixel,
    measured from that cell's original floating-point projected coordinates.

    Output structure (split=True):
        save_dir/
            train/
                region_0.png
                ...
            val/
                region_0.png
                ...

    Output structure (split=False):
        save_dir/
            region_0.png
            region_1.png
            ...

    Args:
        result_df: DataFrame containing region IDs, coordinates, and RGB values.
        save_dir: Root output directory.
        split: Whether to split regions into train/val subdirectories.
        val_ratio: Validation-region fraction (ignored when ``split=False``).
        seed: Random seed for reproducible region splitting.
        region_col: Column name identifying regions.
        x_col: Column name for x-coordinates.
        y_col: Column name for y-coordinates.
        r_col: Column name for red values.
        g_col: Column name for green values.
        b_col: Column name for blue values.
        image_size: Output image height/width.
        filename_prefix: Prefix for saved image filenames.
        nearest_neighbor: If True, resolve pixel collisions by assigning later
            cells to the nearest empty pixel. If False, preserve overwrite
            behavior.

    Returns:
        None. Images are written to disk.
    """
    save_dir = Path(save_dir)

    if split:
        train_dir = save_dir / "train"
        val_dir = save_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        all_regions = list(result_df[region_col].unique())
        rng = random.Random(seed)
        rng.shuffle(all_regions)

        n_val = max(1, int(len(all_regions) * val_ratio))
        val_regions = set(all_regions[:n_val])
        train_regions = set(all_regions[n_val:])
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        all_regions = list(result_df[region_col].unique())
        train_regions = set()
        val_regions = set()

    train_cnt = 0
    val_cnt = 0

    for cnt, reg in enumerate(all_regions):
        subset = result_df[result_df[region_col] == reg]

        xs = subset[x_col].values
        ys = subset[y_col].values
        red_values = subset[r_col].values
        green_values = subset[g_col].values
        blue_values = subset[b_col].values

        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
        occupied = (
            np.zeros((image_size, image_size), dtype=bool)
            if nearest_neighbor
            else None
        )
        skipped_full_image = 0

        for i in range(len(subset)):
            x_int = int(round(xs[i]))
            y_int = int(round(ys[i]))
            if not (0 <= x_int < image_size and 0 <= y_int < image_size):
                continue

            if occupied is not None:
                destination = _nearest_empty_pixel(xs[i], ys[i], occupied)
                if destination is None:
                    skipped_full_image += 1
                    continue
                x_int, y_int = destination
                occupied[y_int, x_int] = True

            img[y_int, x_int, 0] = blue_values[i]
            img[y_int, x_int, 1] = green_values[i]
            img[y_int, x_int, 2] = red_values[i]

        if skipped_full_image:
            warnings.warn(
                f"Region {reg!r}: skipped {skipped_full_image} cell(s) because "
                "the output image has no empty pixels left.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not split:
            save_path = save_dir / f"{filename_prefix}_{cnt}.png"
        elif reg in train_regions:
            save_path = train_dir / f"{filename_prefix}_{train_cnt}.png"
            train_cnt += 1
        else:
            save_path = val_dir / f"{filename_prefix}_{val_cnt}.png"
            val_cnt += 1

        if not cv2.imwrite(str(save_path), img):
            raise OSError(f"Failed to write image: {save_path}")
