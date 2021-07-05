from typing import Optional, Type, Tuple

from dll.activations import _Activation
from dll.layers import BaseLayer
import numpy as np


class MaxPool2d(BaseLayer):
    def __init__(self,
                 kernel_size: int,
                 stride: Optional[int] = None):
        super().__init__(optimize=False)
        self.kernel_size: int = kernel_size
        self.stride: int = kernel_size if stride is None else stride

        self.input_cache: np.ndarray = np.array([])

    def get_num_params(self) -> int:
        return 0

    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        # transpose to (examples, height, width, channels)
        x = x.transpose((0, 2, 3, 1))

        _, h_in, w_in, _ = x.shape

        n_h = 1 + (h_in - self.kernel_size) // self.stride
        n_w = 1 + (w_in - self.kernel_size) // self.stride

        z = np.zeros((x.shape[0], n_h, n_w, x.shape[3]))

        for h in range(n_h):
            for w in range(n_w):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size

                z[:, h, w, :] = np.max(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))

        self.input_cache = x
        # transpose to (examples, channels, height, width)
        return z.transpose((0, 3, 1, 2))

    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # transpose to (examples, height, width, channels)
        grad = grad.transpose((0, 2, 3, 1))
        x = self.input_cache

        m, h_in, w_in, channels = x.shape

        n_h = 1 + (h_in - self.kernel_size) // self.stride
        n_w = 1 + (w_in - self.kernel_size) // self.stride

        da_prev = np.zeros_like(x)

        for h in range(n_h):
            for w in range(n_w):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size

                x_slice = x[:, h_start:h_end, w_start:w_end, :]
                max_arr = np.max(x_slice, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
                mask = x_slice == max_arr

                da_prev[:, h_start:h_end, w_start:w_end, :] = mask * grad[:, h:h+1, w:w+1, :]

        # transpose to (examples, channels, height, width)
        return da_prev.transpose((0, 3, 1, 2)), np.array([]), np.array([])
