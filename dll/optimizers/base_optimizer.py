from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dll import Model


class BaseOptimizer(ABC):
    """A base class for a model optimizer.

    An abstract class containing the `step` method that model optimizers must implement.

    Args:
        model: The model to be optimized.
    """

    def __init__(self, model: 'Model', **kwargs):
        self.model: 'Model' = model

    @abstractmethod
    def step(self, grad: np.ndarray) -> None:
        """Optimizes the model's layers based on the gradient of the loss.

        Args:
            grad: Numpy array of the gradients of the training loss function.
        """
        raise NotImplementedError()
