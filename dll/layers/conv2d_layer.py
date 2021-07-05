from typing import Optional, Type, Tuple

from dll.activations import _Activation
from dll.layers import BaseLayer
import numpy as np


class Conv2d(BaseLayer):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0,
                 padding_mode: Optional[str] = 'zeros',
                 activation: Optional[Type[_Activation]] = None):
        super().__init__()
        self.input_channels: int = input_channels
        self.output_channels: int = output_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding
        self.padding_mode: str = padding_mode
        self.activation: Type[_Activation] = activation

        self.weights: np.ndarray = np.random.randn(self.kernel_size, self.kernel_size, self.input_channels,
                                                   self.output_channels) * 0.1
        self.bias: np.ndarray = np.zeros((self.output_channels,))

        self.input_cache: np.ndarray = np.array([])
        self.activation_cache: np.ndarray = np.array([])

    def get_num_params(self) -> int:
        return self.weights.size + self.bias.size

    def _apply_padding(self, x: np.ndarray) -> np.ndarray:
        return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                      mode='constant', constant_values=(0, 0))

    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        # transpose to (examples, height, width, channels)
        x = x.transpose((0, 2, 3, 1))
        self.input_cache = x

        _, h_in, w_in, _ = x.shape

        n_h = int((h_in - self.kernel_size + 2 * self.padding) / self.stride) + 1
        n_w = int((w_in - self.kernel_size + 2 * self.padding) / self.stride) + 1

        x = self._apply_padding(x)

        z = np.zeros((x.shape[0], n_h, n_w, self.output_channels))

        for h in range(n_h):
            for w in range(n_w):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size

                z[:, h, w, :] = np.sum(x[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                                       self.weights[np.newaxis, :, :, :], axis=(1, 2, 3))

        z = z + self.bias
        # transpose to (examples, channels, height, width)
        z = z.transpose((0, 3, 1, 2))
        self.activation_cache = z
        if self.activation is not None:
            return self.activation.compute(z)
        return z

    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, z = self.input_cache, self.activation_cache

        if self.activation is not None:
            grad = self.activation.backward(grad, z)

        # transpose to (examples, height, width, channels)
        grad = grad.transpose((0, 2, 3, 1))

        m, h_in, w_in, _ = x.shape

        n_h = int((h_in - self.kernel_size + 2 * self.padding) / self.stride) + 1
        n_w = int((w_in - self.kernel_size + 2 * self.padding) / self.stride) + 1

        x = self._apply_padding(x)

        da_prev = np.zeros_like(self.input_cache)
        dw = np.zeros_like(self.weights)
        db = np.sum(grad, axis=(0, 1, 2))

        for h in range(n_h):
            for w in range(n_w):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size

                da_prev[:, h_start:h_end, w_start:w_end, :] += np.sum(self.weights[np.newaxis, :, :, :, :] *
                                                                     grad[:, h:h+1, w:w+1, np.newaxis, :], axis=4)

                dw += np.sum(x[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                             grad[:, h:h+1, w:w+1, np.newaxis, :], axis=0)

        dw /= m
        da_prev = da_prev[:, self.padding:self.padding + h_in, self.padding:self.padding + w_in, :]

        # transpose to (examples, channels, height, width)
        return da_prev.transpose((0, 3, 1, 2)), dw, db
