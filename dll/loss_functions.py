from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class _LossFunction(ABC):
    @staticmethod
    @abstractmethod
    def compute_cost(y_pred: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError()


class BCELoss(_LossFunction):
    @staticmethod
    def compute_cost(a: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / y.shape[0]
        grad = -(np.divide(y, a) - np.divide(1 - y, 1 - a))
        return np.squeeze(cost), grad


class MSELoss(_LossFunction):
    @staticmethod
    def compute_cost(y_pred: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        cost = np.mean(np.power(y - y_pred, 2))
        grad = 2 * (y_pred - y) / y.shape[0]
        return np.squeeze(cost), grad


class CrossEntropyLoss(_LossFunction):
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @staticmethod
    def compute_cost(y_pred: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        # print(y.shape, y)
        m = y.shape[0]
        sm_output = CrossEntropyLoss._softmax(y_pred)
        log_likelihood = -np.log(sm_output[range(m), y])

        cost = np.sum(log_likelihood) / m
        grad = sm_output
        grad[range(m), y] -= 1
        return np.squeeze(cost), grad / m