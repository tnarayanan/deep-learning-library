from abc import ABC, abstractmethod

import numpy as np


class _Activation(ABC):
    @staticmethod
    @abstractmethod
    def compute(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Sigmoid(_Activation):
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        grad = Sigmoid.compute(z) * (1 - Sigmoid.compute(z))
        return da * grad


class ReLU(_Activation):
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        grad = (z > 0).astype(np.float32)
        return da * grad


class Tanh(_Activation):
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
        return da * (1 - np.tanh(z) ** 2)
