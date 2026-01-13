"""Loss functions."""

from .base import Layer


class MSELoss(Layer):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        self.y_pred = y_pred
        self.y_true = y_true
        return 1 / 2 * (y_pred - y_true) ** 2

    def backward(self):
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError("backward() called before forward()")

        # dL/dy_pred = d(1/2 * (y_pred - y_true)^2) / dy_pred
        #            = 1/2 * 2 * (y_pred - y_true)
        #            = y_pred - y_true
        return self.y_pred - self.y_true
