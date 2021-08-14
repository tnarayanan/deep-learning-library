import numpy as np

from dll.optimizers import BaseOptimizer


class SGD(BaseOptimizer):
    """A stochastic gradient descent (SGD) optimizer.

    Args:
        model: The model to be optimized.
        learning_rate: A float representing the learning rate of the optimizer.
    """

    def __init__(self, model, learning_rate: float):
        super().__init__(model)
        self.learning_rate = learning_rate

    def step(self, grad: np.ndarray) -> None:
        da_prev = grad
        for layer in reversed(self.model.layers):
            da_prev, dw, db = layer.backward(da_prev)
            if layer.optimize:
                layer.weights = layer.weights - self.learning_rate * dw
                layer.bias = layer.bias - self.learning_rate * db
