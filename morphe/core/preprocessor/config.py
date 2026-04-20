from dataclasses import dataclass


@dataclass(frozen=True)
class PreProcessConfig:
    """
    Configuration for preprocessing the input dataframe before training.
    
    region_col: Name of the column containing region identifiers. Preprocessing is done separately for each region.
    pos_cols: Tuple of (x_col, y_col) names for the coordinate columns. 
    original_dimensions: Optional tuple of (max_x, max_y) specifying the original coordinate extent for rescaling. If None, computed from the data at preprocess time.
    """
    
    region_col: str = "unique_region"
    pos_cols: tuple[str, str] = ("x", "y")
    original_dimensions: tuple[int, int] | None = None