import numpy as np


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray):
    return np.max(0, x)
