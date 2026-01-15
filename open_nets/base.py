"""Base classes."""

from typing import Any, Protocol


class Layer(Protocol):
    """Base class for neural network layers."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the class."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass, to be implemented by subclasses."""
        pass
