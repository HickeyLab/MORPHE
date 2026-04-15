from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


from pathlib import Path
import os
import random

import cv2
import numpy as np
import pandas as pd


def rasterize_rgb_regions(
    result_df: pd.DataFrame,
    save_dir: str | Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
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
    Rasterize per-region RGB data into train/val image folders.

    Each unique region is assigned entirely to either the train or validation
    split, then rasterized into a PNG image.

    Output structure:
        save_dir/
            train/
                region_0.png
                region_1.png
                ...
            val/
                region_0.png
                region_1.png
                ...

    Args:
        result_df: DataFrame containing region IDs, coordinates, and RGB values.
        save_dir: Root output directory.
        val_ratio: Fraction of regions to place in validation split.
        seed: Random seed for reproducible region splitting.
        region_col: Column name identifying regions.
        x_col: Column name for x-coordinates.
        y_col: Column name for y-coordinates.
        r_col: Column name for red values.
        g_col: Column name for green values.
        b_col: Column name for blue values.
        image_size: Output image height/width.
        filename_prefix: Prefix for saved image filenames.

    Returns:
        None. Images are written to disk.
    """
    save_dir = Path(save_dir)
    train_dir = save_dir / "train"
    val_dir = save_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    all_regions = list(result_df[region_col].unique())
    rng = random.Random(seed)
    rng.shuffle(all_regions)

    n_val = int(len(all_regions) * val_ratio)
    val_regions = set(all_regions[:n_val])
    train_regions = set(all_regions[n_val:])

    train_cnt = 0
    val_cnt = 0

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

        if reg in train_regions:
            save_path = train_dir / f"{filename_prefix}_{train_cnt}.png"
            train_cnt += 1
        else:
            save_path = val_dir / f"{filename_prefix}_{val_cnt}.png"
            val_cnt += 1

        cv2.imwrite(str(save_path), img)