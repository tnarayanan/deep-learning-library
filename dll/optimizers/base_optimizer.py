from abc import ABC, abstractmethod

import numpy as np
from dll import Model


class BaseOptimizer(ABC):
    def __init__(self, model: Model):
        self.model: Model = model

    @abstractmethod
    def step(self, grad: np.ndarray) -> None:
        raise NotImplementedError()
