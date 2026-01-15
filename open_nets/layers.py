"""Linear layers"""

import numpy as np

from .base import Layer


class Linear(Layer):
    """Linear layer."""

    def __init__(self, input_size: int, output_size: int, random_init: bool = True, verbose: bool = False) -> None:
        """Initialize layer."""
        self.w = np.random.randn(input_size, output_size) if random_init else np.zeros((input_size, output_size))
        self.b = np.random.randn(output_size) if random_init else np.zeros(output_size)
        self.x = None

        if verbose:
            print(f"weights: {self.w}")
            print(f"biases: {self.b}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.x = x
        # z = x @ w + b
        return np.dot(x, self.w) + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        if self.x is None:
            raise RuntimeError("backward() called before forward()")

        # dz/dw = d(w * x + b) / dw
        #       = x^T @ grad
        grad_w = np.dot(self.x.T, grad)

        # dz/db = d(w * x + b) / db
        #       = sum(grad, axis=0)
        grad_b = np.sum(grad, axis=0)

        # dz/dx = d(w * x + b) / dx
        #       = grad @ w
        grad_x = np.dot(grad, self.w.T)

        return grad_w, grad_b, grad_x
