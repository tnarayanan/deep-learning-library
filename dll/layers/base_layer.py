from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseLayer(ABC):
    def __init__(self, optimize: Optional[bool] = True):
        self.optimize = optimize

    def get_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get_num_params(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, True)
