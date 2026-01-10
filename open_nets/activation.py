"""Activation layers."""

import numpy as np

from .base import Layer


class Sigmoid(Layer):
    def __init__(self):
        self.s = None

    def forward(self, x):
        self.s = 1 / (1 + np.exp(-x))
        return self.s

    def backward(self, grad):
        if self.s is None:
            raise RuntimeError("backward() called before forward()")

        # d(sigmoid)/dx = s * (1 - s)
        return grad * self.s * (1 - self.s)
