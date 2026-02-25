from matplotlib import pyplot as plt


from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_decoded_image(
    *,
    preview: Sequence[float],
    iteration: int | None = 0,
    figsize: tuple[int, int] = (6, 6),
    title: str | None = None,
) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(preview)
    plt.title(title if title else f"Iteration {iteration+1}")
    plt.axis("off")
    plt.show()


def plot_inpainting_triplet(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    figsize: tuple[int, int] = (14, 4),
    title_prefix: str | None = None,
) -> None:

    def t(name: str) -> str:
        return f"{title_prefix} {name}" if title_prefix else name

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.title(t("Input"))
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(t("Mask"))
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(t("Inpaint Output"))
    plt.imshow(result)
    plt.axis("off")

    plt.tight_layout()
    plt.show()