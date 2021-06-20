from dll.layers import BaseLayer
import dll

import numpy as np


class Model(object):
    def __init__(self, layers: list[BaseLayer]):
        self.layers: list[BaseLayer] = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = dll.sigmoid(layer(x))

        return x
