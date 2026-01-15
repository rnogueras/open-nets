"""Loss functions."""

import numpy as np

from .base import Layer


class MSELoss(Layer):
    """Mean Squared Error (MSE) loss function."""

    def __init__(self) -> None:
        """Initialize instance."""
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Forward pass."""
        assert y_pred.shape == y_true.shape
        self.y_pred = y_pred
        self.y_true = y_true
        return 1 / 2 * (y_pred - y_true) ** 2

    def backward(self) -> np.ndarray:
        """Backward pass."""
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("backward() called before forward()")

        # dL/dy_pred = d(1/2 * (y_pred - y_true)^2) / dy_pred
        #            = 1/2 * 2 * (y_pred - y_true)
        #            = y_pred - y_true
        return self.y_pred - self.y_true
