from matplotlib import pyplot as plt
import numpy as np


def save_side_by_side_barplot(
        self,
        xs: np.ndarray,
        left_vals: np.ndarray,
        right_vals: np.ndarray,
        left_title: str,
        right_title: str,
        out_path: str,
    ):
        if xs.ndim != 1:
            raise ValueError("xs must be 1D")
        if left_vals.shape != xs.shape or right_vals.shape != xs.shape:
            raise ValueError("bar values must match xs shape")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(xs, left_vals)
        axes[0].set_title(left_title)

        axes[1].bar(xs, right_vals)
        axes[1].set_title(right_title)

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()