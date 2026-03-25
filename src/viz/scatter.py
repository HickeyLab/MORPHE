from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

from disco.viz.palettes import DEFAULT_CELL_TYPE_COLORS

def show_cell_region_scatterplot(
    df: pd.DataFrame,
    *,
    region_name: str,
    x: str = "x",
    y: str = "y",
    cell_type_col: str = "Cell Type",
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (10, 10),
    alpha: float = 0.5,
    s: float = 6,
    show: bool = True,
) -> Axes:   
    # Validate region name
    region_df = df.groupby("unique_region").get_group(region_name)
    if region_df.empty:
        raise ValueError(f"Region '{region_name}' not found")
    
    # Validate cell type colors
    if palette is not None and not isinstance(palette, dict):
        raise TypeError("palette must be a dict mapping between cell type and color.")
    
    # Determine cell type colors
    palette = DEFAULT_CELL_TYPE_COLORS if palette is None else palette

    # Plot figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=region_df,
        x=x,
        y=y,
        hue=cell_type_col,
        palette=palette,
        alpha=alpha,
        s=s,
        ax=ax,
    )

    ax.invert_yaxis()
    ax.set_title(f"Scatter Plot for Region: {region_name}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title=cell_type_col, prop={"size": 6})

    # Allow user to determine whether to show the figure
    if show:
        plt.show()

    return ax