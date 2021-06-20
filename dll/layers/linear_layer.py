from dll.layers import BaseLayer
import numpy as np


class Linear(BaseLayer):
    def __init__(self, input_units: int, output_units: int):
        self.input_units: int = input_units
        self.output_units: int = output_units

        self.weights: np.ndarray = np.random.randn(self.output_units, self.input_units) * 0.01
        self.bias: np.ndarray = np.zeros((self.output_units, 1))

        self.linear_cache = None
        self.activation_cache = None

    def get_num_params(self) -> int:
        return self.weights.size + self.bias.size

    def get_output_shape(self) -> tuple:
        return self.output_units, 1

    def forward(self, x: np.ndarray, is_training: bool) -> np.ndarray:
        z = np.dot(self.weights, x) + self.bias
        self.linear_cache = (x, self.weights, self.bias)
        self.activation_cache = z
        return z

    def backward(self, curr_grad):
        pass
