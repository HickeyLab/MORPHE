from dataclasses import dataclass


@dataclass(frozen=True)
class PreProcessConfig:
    region_col: str = "unique_region"
    pos_cols: tuple[str, str] = ("x", "y")
    original_dimensions: tuple[int, int] | None = None
    """
    Maximum (x, y) coordinate extent of the dataset, used as the upper bound
    for coordinate rescaling. If None (default), auto-computed from the data
    as (max(x_col), max(y_col)) at preprocess time. Override only when you
    need coordinates to be consistent across splits of the same slide (e.g.,
    training on a subset but wanting the same grid as the full slide).
    """