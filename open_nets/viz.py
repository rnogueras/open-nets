"""Visualization module."""

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def plot(
    *functions: Callable,
    samples: np.ndarray | None = None,
    x_min: float = -2,
    x_max: float = 2,
    num_points: int = 1000,
    ax: plt.Axes | None = None,
    title=None,
) -> plt.Axes:
    """Plot simple or composite function."""
    x = np.linspace(x_min, x_max, num_points)[:, None]
    y = x.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if samples is not None:
        preds = samples.copy()
        for function in functions:
            preds = function(preds)
        ax.scatter(samples, preds, color="orange", linewidth=2, label="Samples")

    for function in functions:
        y = function(y)

    ax.plot(x, y, linewidth=2, label="Model")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if title:
        ax.set_title(title)  # Main title with some padding

    plt.tight_layout()
    plt.show()
    return ax
