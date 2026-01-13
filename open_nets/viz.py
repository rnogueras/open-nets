"""Visualization module."""

import numpy as np
import matplotlib.pyplot as plt


def plot(*functions, x_min=-2, x_max=2, num_points=1000, color="b-", ax=None):
    x = np.linspace(x_min, x_max, num_points)[:, None]
    z = x.copy()

    for f in functions:
        z = f(z)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, z, color, linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    return ax
