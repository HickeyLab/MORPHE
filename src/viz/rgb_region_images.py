from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def save_region_rgb_images(
    df: pd.DataFrame,
    *,
    save_dir: Path,
    image_size: int = 1024,
    region_col: str = "region",
    x_col: str = "x",
    y_col: str = "y",
    rgb_cols: tuple[str, str, str] = ("R", "G", "B"),
) -> None:
    # Basic type validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df is empty.")

    if not isinstance(save_dir, (str, Path)):
        raise TypeError("save_dir must be a str or pathlib.Path.")
    save_dir = Path(save_dir)

    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")

    for name, val in [
        ("region_col", region_col),
        ("x_col", x_col),
        ("y_col", y_col),
    ]:
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    if (
        not isinstance(rgb_cols, tuple)
        or len(rgb_cols) != 3
        or not all(isinstance(c, str) and c.strip() for c in rgb_cols)
    ):
        raise ValueError('rgb_cols must be a 3-tuple of non-empty strings, e.g. ("R","G","B").')

    # Required column validation
    required = [region_col, x_col, y_col, *rgb_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")
    
    r = df[rgb_cols[0]].to_numpy()
    g = df[rgb_cols[1]].to_numpy()
    b = df[rgb_cols[2]].to_numpy()

    rgb = np.stack([r, g, b], axis=1)

    # Catch NaN/inf and non-numeric early
    if not np.issubdtype(rgb.dtype, np.number):
        raise TypeError(f"RGB columns {rgb_cols} must be numeric.")
    if not np.isfinite(rgb).all():
        raise ValueError(f"RGB columns {rgb_cols} contain NaN/inf.")

    # Range check (works correctly now that NaN/inf are excluded)
    if (rgb < 0).any() or (rgb > 255).any():
        raise ValueError(
            f"RGB columns {rgb_cols} must be in [0, 255]. "
            "If your values are floats or scaled, normalize/cast first."
        )

    # Save file
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Could not create/access save_dir={save_dir!s}: {e}") from e

    all_regions = df[region_col].unique()
    cnt = 0
    for reg in all_regions:
        subset = df[df[region_col] == reg]
        xs = subset[x_col].values
        ys = subset[y_col].values
        Rs = subset[rgb_cols[0]].values
        Gs = subset[rgb_cols[1]].values
        Bs = subset[rgb_cols[2]].values

        # Create a blank image (adjust size if needed)
        # Example: 1024 x 1024, 3 channels
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        for i in range(len(subset)):
            x_int = int(round(xs[i]))
            y_int = int(round(ys[i]))
            # Safety check
            if 0 <= x_int < image_size and 0 <= y_int < image_size:
                img[y_int, x_int] = (int(Bs[i]), int(Gs[i]), int(Rs[i]))

        # Save the image
        save_path = save_dir / f"region_{cnt}.png"
        cv2.imwrite(str(save_path), img)
        cnt += 1

        print(f"Saved image for region {cnt} -> {save_path}")
