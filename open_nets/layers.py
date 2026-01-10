"""Linear layers"""

import numpy as np

from .base import Layer


class Linear(Layer):
    def __init__(self, random_init=True, verbose=True):
        self.w = np.random.randn() if random_init else 0
        self.b = np.random.randn() if random_init else 0
        self.x = None

        if verbose:
            print(f"weight: {self.w}")
            print(f"bias: {self.b}")

    def forward(self, x):
        self.x = x
        # z = w * x + b
        return self.w * x + self.b

    def backward(self, grad):
        if self.x is None:
            raise RuntimeError("backward() called before forward()")

        # dz/dw = d(w * x) / dw + d(b) / dw
        #       = d(w * x) / dw + 0
        #       = x * d(w) / dw + 0
        #       = x * 1 + 0
        #       = x
        grad_w = self.x * grad

        # dz/dx = d(w * x + b) / dx
        #       = d(w * x) / dx + d(b) / dx
        #       = d(w * x) / dx + 0
        #       = w * d(x) / dx + 0
        #       = w * 1 + 0
        #       = w
        grad_x = grad * self.w

        # dz/db = d(w * x + b) / db
        #       = d(w * x) / db + d(b) / db
        #       = 0 + d(b) / db
        #       = 0 + 1
        #       = 1
        grad_b = grad * 1

        return grad_w, grad_b, grad_x
