from abc import ABC, abstractmethod
import numpy as np


class _CostFunction(ABC):
    @staticmethod
    @abstractmethod
    def compute_cost(y_pred: np.ndarray, y: np.ndarray):
        return NotImplementedError()


class CrossEntropy(_CostFunction):
    @staticmethod
    def compute_cost(a: np.ndarray, y: np.ndarray):
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / y.shape[-1]
