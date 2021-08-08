from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dll import Model


class BaseOptimizer(ABC):
    def __init__(self, model: 'Model', **kwargs):
        self.model: 'Model' = model

    @abstractmethod
    def step(self, grad: np.ndarray) -> None:
        raise NotImplementedError()
