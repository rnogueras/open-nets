"""Activation layers."""

import numpy as np

from .base import Layer


class Sigmoid(Layer):
    """Sigmoid activation."""

    def __init__(self) -> None:
        """Initialize class."""
        self.s = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.s = 1 / (1 + np.exp(-x))
        return self.s

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        if self.s is None:
            raise RuntimeError("backward() called before forward()")

        # d(sigmoid)/dx = s * (1 - s)
        return grad * self.s * (1 - self.s)


class ReLU(Layer):
    """ReLU activation."""

    def __init__(self):
        """Initialize class."""
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        if self.mask is None:
            raise RuntimeError("backward() called before forward()")

        # d(ReLU)/dx = 1 for positive x, and 0 for negative x
        return grad * self.mask
