from typing import Optional

from dll.layers import BaseLayer
import numpy as np


class Flatten(BaseLayer):
    def __init__(self, input_shape: tuple):
        super().__init__(optimize=False)
        self.input_shape: tuple = input_shape
        self.output_units: int = np.prod(input_shape)

    def get_num_params(self) -> int:
        return 0

    def get_input_shape(self) -> tuple:
        return (np.newaxis,) + self.input_shape

    def get_output_shape(self) -> tuple:
        return np.newaxis, self.output_units  # np.newaxis represents the batch size

    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        return x.reshape((-1, self.output_units))

    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return grad.reshape((-1,) + self.input_shape), np.array([]), np.array([])
