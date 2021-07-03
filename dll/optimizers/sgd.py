import numpy as np

from dll.optimizers import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, model, learning_rate):
        super().__init__(model)
        self.learning_rate = learning_rate

    def step(self, grad: np.ndarray) -> None:
        da_prev = grad
        for layer in reversed(self.model.layers):
            da_prev, dw, db = layer.backward(da_prev)
            layer.weights = layer.weights - self.learning_rate * dw
            layer.bias = layer.bias - self.learning_rate * db
