from typing import Iterable

import pandas as pd


def compute_region_density(
    grouped_by_region: Iterable[tuple[str, pd.DataFrame]]
) -> dict[str, float]:
    ratios = {}
    for region, region_data in grouped_by_region:
        max_x = region_data["x"].max()
        max_y = region_data["y"].max()
        total_possible = (max_x + 1) * (max_y + 1)
        ratios[region] = (
            len(region_data) / total_possible
            if total_possible > 0 else 0
        )
    return ratios