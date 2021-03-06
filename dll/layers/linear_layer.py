from typing import Optional, Type

import numpy as np

from dll.activations import _Activation
from dll.layers import BaseLayer


class Linear(BaseLayer):
    """A linear (fully-connected) layer.

    Args:
        input_units: The number of inputs to the layer.
        output_units: The number of outputs of the layer.
        activation: Optional; the activation function of the layer.
    """

    def __init__(self, input_units: int, output_units: int, activation: Optional[Type[_Activation]] = None):
        super().__init__()
        self.input_units: int = input_units
        self.output_units: int = output_units
        self.activation: Optional[Type[_Activation]] = activation

        self.weights: np.ndarray = np.random.randn(self.input_units, self.output_units) * 0.01
        self.bias: np.ndarray = np.zeros((self.output_units,))

        self.input_cache: np.ndarray = np.array([])
        self.activation_cache: np.ndarray = np.array([])

    def get_num_params(self) -> int:
        return self.weights.size + self.bias.size

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
        assert(isinstance(db, np.ndarray))
        da_prev = np.dot(dz, self.weights.T)

        return da_prev, dw, db
