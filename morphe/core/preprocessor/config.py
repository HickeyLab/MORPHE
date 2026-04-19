from dataclasses import dataclass


@dataclass(frozen=True)
class PreProcessConfig:
    original_dimensions: tuple[int, int] = (9406, 9070)
    region_col: str = "unique_region"
    pos_cols: tuple[str, str] = ("x", "y")