import math
from typing import Optional

import numpy as np

from dll.layers import BaseLayer


class Flatten(BaseLayer):
    """A layer that flattens the input.

    Args:
        input_shape: The shape of the input to the layer.
    """

    def __init__(self, input_shape: tuple):
        super().__init__(optimize=False)
        self.input_shape: tuple = input_shape
        self.output_units: int = math.prod(input_shape)

    def get_num_params(self) -> int:
        return 0

    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        return x.reshape((-1, self.output_units))

    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return grad.reshape((-1,) + self.input_shape), np.array([]), np.array([])
