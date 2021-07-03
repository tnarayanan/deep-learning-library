from typing import Optional, Type

from dll.activations import _Activation
from dll.layers import BaseLayer
import numpy as np


class Linear(BaseLayer):
    def __init__(self, input_units: int, output_units: int, activation: Optional[Type[_Activation]] = None):
        self.input_units: int = input_units
        self.output_units: int = output_units
        self.activation: Type[_Activation] = activation

        self.weights: np.ndarray = np.random.randn(self.input_units, self.output_units) * 0.01
        self.bias: np.ndarray = np.zeros((self.output_units,))

        self.input_cache: np.ndarray = np.array([])
        self.activation_cache: np.ndarray = np.array([])

    def get_num_params(self) -> int:
        return self.weights.size + self.bias.size

    def get_input_shape(self) -> tuple:
        return np.newaxis, self.input_units

    def get_output_shape(self) -> tuple:
        return np.newaxis, self.output_units  # np.newaxis represents the batch size

    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        z = np.dot(x, self.weights) + self.bias

        self.input_cache = x
        self.activation_cache = z

        if self.activation is not None:
            return self.activation.compute(z)
        return z

    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # activation gradient
        x, z = self.input_cache, self.activation_cache

        if self.activation is not None:
            dz = self.activation.backward(grad, z)
        else:
            dz = grad

        dw = np.dot(x.T, dz)
        db = np.sum(dz, axis=0, keepdims=True).squeeze()
        da_prev = np.dot(dz, self.weights.T)

        return da_prev, dw, db
