from abc import ABC, abstractmethod

import numpy as np


class _Activation(ABC):
    """A base class for an activation function.

    An abstract class containing the methods that activation functions must implement.
    """
    @staticmethod
    @abstractmethod
    def compute(x: np.ndarray) -> np.ndarray:
        """Computes the activation function for the given input.

        Args:
            x: A Numpy array representing the input to the activation function.

        Returns:
            The output of the activation function.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the activation function.

        Args:
            da: The activation gradient of the next layer in the model
            z: The cached outputs of the current layer before the activation function was applied.

        Returns:
            The gradient of the activation function.
        """
        raise NotImplementedError()


class Sigmoid(_Activation):
    """The sigmoid activation function.
    """
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        grad = Sigmoid.compute(z) * (1 - Sigmoid.compute(z))
        return da * grad


class ReLU(_Activation):
    """The ReLU activation function.
    """
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        grad = (z > 0).astype(np.float32)
        return da * grad


class Tanh(_Activation):
    """The Tanh activation function.
    """
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        return da * (1 - np.tanh(z) ** 2)
