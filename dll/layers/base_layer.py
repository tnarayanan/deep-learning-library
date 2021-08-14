from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseLayer(ABC):
    """A base class for a neural network layer.

    An abstract class containing methods that neural network layers must implement.

    Args:
        optimize: A boolean representing whether to optimize the layer's parameters.
    """

    def __init__(self, optimize: bool = True):
        self.optimize = optimize
        if self.optimize:
            self.weights: np.ndarray = np.array([])
            self.bias: np.ndarray = np.array([])

    def get_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get_num_params(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        """Executes a forward-pass of the layer with the given input.

        Args:
            x: A Numpy array representing the input to the layer.
            is_training: A boolean representing whether the forward pass occurs during model training or evaluation.

        Returns:
            The output of the forward-pass of the layer.
        """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates the gradients of the layer.

        Args:
            grad: The gradient of the next layer in the model.

        Returns:
            The activation, weights, and bias gradients of the layer.
        """
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, True)
