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
    region_col: str = "region",
    x_col: str = "x",
    y_col: str = "y",
    r_col: str = "R",
    g_col: str = "G",
    b_col: str = "B",
    image_size: int = 1024,
    filename_prefix: str = "region",
) -> None:
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