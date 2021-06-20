from abc import ABC, abstractmethod
import numpy as np


class BaseLayer(ABC):
    def get_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get_num_params(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_output_shape(self) -> tuple:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: np.ndarray, is_training: bool) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, curr_grad):
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, True)
