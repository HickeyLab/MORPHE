import copy

import pandas as pd

from .config import PreProcessConfig


class PreProcessor:
    """
    Preprocess spatial cell data according to a preprocessing configuration.

    This class currently supports reducing the coordinate grid dimensions while
    preserving unique cell positions within each region.
    """

    def __init__(self, config: PreProcessConfig) -> None:
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration containing the original grid
                dimensions and any other preprocessing settings.
        """
        self.config = config

    def _calculate_dimensions(
        self,
        passing: tuple[int, int],
        failing: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Compute the midpoint dimensions between a known passing size and a known
        failing size.

        If the midpoint collapses to the failing dimensions, the passing
        dimensions are returned to terminate the binary-search-like reduction.

        Args:
            passing: Most recent dimensions known to succeed without coordinate
                conflicts.
            failing: Most recent dimensions known to fail due to coordinate
                collisions.

        Returns:
            The next dimensions to test.
        """
        x_p, y_p = passing
        x_f, y_f = failing

        x_new = round((x_p + x_f) * 0.5)
        y_new = round((y_p + y_f) * 0.5)

        new = (x_new, y_new)

        if new == failing:
            return passing

        return new

    def _compute_new_coord_pair(
        self,
        coord_pair: tuple[float, float],
        dimensions: tuple[int, int],
        original_dimensions: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Scale a coordinate pair from the original grid into a reduced grid.

        Args:
            coord_pair: Original `(x, y)` coordinate pair.
            dimensions: Target grid dimensions.
            original_dimensions: Original grid dimensions.

        Returns:
            The rescaled coordinate pair as integer grid coordinates.
        """
        c_1, c_2 = coord_pair
        x, y = dimensions
        x_i, y_i = original_dimensions

        x_factor = x / x_i
        y_factor = y / y_i

        return (round(c_1 * x_factor), round(c_2 * y_factor))

    def _reduce_dimensions(
        self,
        df_initial: pd.DataFrame,
        original_dimensions: tuple[int, int],
        region_col: str = "unique_region",
        x_col: str = "x",
        y_col: str = "y",
        *,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, tuple[int, int]]:
        """
        Reduce grid dimensions as much as possible without introducing
        coordinate collisions within any region.

        The method performs a binary-search-like procedure over possible grid
        sizes. For each candidate size, all cell coordinates are rescaled and
        checked for uniqueness within each region. If a collision is found, the
        candidate dimensions are treated as failing; otherwise, they are treated
        as passing.

        Args:
            df_initial: Input dataframe containing at least `unique_region`,
                `x`, and `y` columns.
            original_dimensions: Original grid dimensions used as the starting
                upper bound for reduction.
            verbose: Whether to print progress information during reduction.

        Returns:
            A tuple containing:
                - A dataframe with reduced `x` and `y` coordinates.
                - The optimal reduced dimensions.
        """
        failing_dimensions = (1, 1)
        passing_dimensions = original_dimensions
        dimensions = failing_dimensions
        restart_outer_loop = True
        df = copy.deepcopy(df_initial)

        while dimensions != passing_dimensions:
            if restart_outer_loop:
                failing_dimensions = dimensions
            else:
                passing_dimensions = dimensions

            dimensions = self._calculate_dimensions(
                passing_dimensions,
                failing_dimensions,
            )
            if verbose:
                print(f"Current dimensions: {dimensions}")

            df = copy.deepcopy(df_initial)
            grouped_by_region = df.groupby(region_col)
            restart_outer_loop = False

            coords: set[tuple[int, int]] = set()
            for region_name, region_data in grouped_by_region:
                coords.clear()

                for idx, row in region_data.iterrows():
                    x = row[x_col]
                    y = row[y_col]
                    coord_pair = (x, y)

                    new_coord_pair = self._compute_new_coord_pair(
                        coord_pair,
                        dimensions,
                        original_dimensions,
                    )

                    if new_coord_pair in coords and coord_pair != (4915.0, 7055.0):
                        if verbose:
                            print(
                                f"failing coordinate pair: {new_coord_pair}, "
                                f"scaled from {coord_pair} in region {region_name}"
                            )
                        restart_outer_loop = True
                        break

                    x_n, y_n = new_coord_pair
                    df.at[idx, x_col] = x_n
                    df.at[idx, y_col] = y_n
                    coords.add(new_coord_pair)

                if restart_outer_loop:
                    break

                if verbose:
                    print(f"successfully reduced region {region_name}")

        return df, dimensions

    def preprocess(
        self,
        df: pd.DataFrame,
        *,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, tuple[int, int]]:
        """
        Run preprocessing on the input dataframe.

        Args:
            df: Input dataframe to preprocess.
            verbose: Whether to print progress information during preprocessing.

        Returns:
            A tuple containing:
                - The preprocessed dataframe.
                - The reduced grid dimensions.
        """
        new_df, new_dimensions = self._reduce_dimensions(
            df,
            self.config.original_dimensions,
            region_col=self.config.region_col,
            x_col=self.config.pos_cols[0],
            y_col=self.config.pos_cols[1],
            verbose=verbose,
        )
        return new_df, new_dimensions
