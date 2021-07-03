from dll.layers import BaseLayer

from typing import Sequence, Optional
import numpy as np


class Model(BaseLayer):
    def __init__(self, layers: Sequence[BaseLayer]):
        self.layers: Sequence[BaseLayer] = layers

    def get_num_params(self) -> int:
        total_params = 0
        for layer in self.layers:
            total_params += layer.get_num_params()
        return total_params

    def get_input_shape(self) -> tuple:
        return self.layers[0].get_input_shape()

    def get_output_shape(self) -> tuple:
        return self.layers[-1].get_output_shape()

    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        raise AssertionError("Cannot call function `backward` on a Model instance.")

    def print_summary(self) -> None:
        left_space_len = 2
        layer_padding = 12
        output_shape_padding = 15
        params_padding = 8
        right_space_len = 2

        left_space = " " * left_space_len
        right_space = " " * right_space_len

        def print_line(layer: str, shape: str, params: str):
            print(
                f'{left_space}{layer: >{layer_padding - 3}}   {shape: <{output_shape_padding - 2}}  {params: >{params_padding}}{right_space}')

        thick_line = "=" * (left_space_len + layer_padding + output_shape_padding + params_padding + right_space_len)
        thin_line = "—" * (left_space_len + layer_padding + output_shape_padding + params_padding + right_space_len)

        print_line("Layer", "Output Shape", "# Params")
        print(thick_line)
        print_line("Input", repr(self.layers[0].get_input_shape()), "")

        for layer in self.layers:
            print(thin_line)
            print_line(layer.get_name(), repr(layer.get_output_shape()), str(layer.get_num_params()))

        print(thick_line)
        print(f'{left_space}Total params: {self.get_num_params()}')
        print()
