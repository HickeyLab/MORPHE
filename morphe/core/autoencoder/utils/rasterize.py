from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def rasterize_rgb_regions(
    result_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    region_col: str = "unique_region",
    x_col: str = "x",
    y_col: str = "y",
    r_col: str = "R",
    g_col: str = "G",
    b_col: str = "B",
    image_size: int = 1024,
    filename_prefix: str = "region",
) -> None:
    """
    Rasterize per-region RGB data from a dataframe into image files.

    This function groups rows in ``result_df`` by a region identifier and
    generates one RGB image per region. Each row represents a single pixel
    location with associated RGB values. Pixels are placed into an
    ``image_size x image_size`` canvas, with unspecified pixels initialized
    to white.

    Notes:
        - Coordinates are rounded to the nearest integer pixel location.
        - RGB values are written in OpenCV BGR order when saving.
        - Pixels falling outside the image bounds are ignored.

    Args:
        result_df: DataFrame containing at minimum:
            - a region identifier column,
            - x/y coordinate columns,
            - RGB value columns.
        save_dir: Directory where output images will be saved. Created if it
            does not exist.
        region_col: Column name identifying regions. Each unique value produces
            a separate output image.
        x_col: Column name for x-coordinates (horizontal axis).
        y_col: Column name for y-coordinates (vertical axis).
        r_col: Column name for red channel values.
        g_col: Column name for green channel values.
        b_col: Column name for blue channel values.
        image_size: Height and width (in pixels) of the output square images.
        filename_prefix: Prefix used when naming output files. Files are saved
            as ``{filename_prefix}_{i}.png``.

    Returns:
        None. Images are written to disk.

    Raises:
        KeyError: If required columns are missing from ``result_df``.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_regions = result_df[region_col].unique()
    cnt = 0
    for reg in all_regions:
        subset = result_df[result_df[region_col] == reg]
        xs = subset[x_col].values
        ys = subset[y_col].values
        Rs = subset[r_col].values
        Gs = subset[g_col].values
        Bs = subset[b_col].values

        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        for i in range(len(subset)):
            x_int = int(round(xs[i]))
            y_int = int(round(ys[i]))
            if 0 <= x_int < image_size and 0 <= y_int < image_size:
                img[y_int, x_int, 0] = Bs[i]
                img[y_int, x_int, 1] = Gs[i]
                img[y_int, x_int, 2] = Rs[i]

        save_path = os.path.join(save_dir, f"{filename_prefix}_{cnt}.png")
        cv2.imwrite(save_path, img)
        cnt += 1
