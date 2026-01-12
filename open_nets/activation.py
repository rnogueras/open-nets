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


class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad):
        if self.mask is None:
            raise RuntimeError("backward() called before forward()")

        # d(ReLU)/dx = 1 for positive x, and 0 for negative x
        return grad * self.mask
