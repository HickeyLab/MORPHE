from matplotlib import pyplot as plt


from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_loss_curve(
    *,
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    figsize: tuple[int, int] = (7, 5),
    title: str = "Loss Curve",
    save_path: str | Path | None = None,
) -> Figure:
    # Validate
    if not isinstance(train_losses, Sequence) or len(train_losses) == 0:
        raise ValueError("train_losses must be a non-empty sequence.")

    if val_losses is not None:
        if not isinstance(val_losses, Sequence):
            raise TypeError("val_losses must be a sequence or None.")
        if len(val_losses) != len(train_losses):
            raise ValueError("val_losses must have same length as train_losses.")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="train")

    if val_losses is not None:
        ax.plot(epochs, val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig