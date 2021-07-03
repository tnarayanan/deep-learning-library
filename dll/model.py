from dll.layers import BaseLayer

from typing import Sequence, Optional, Type
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dll.loss_functions import _LossFunction, CrossEntropyLoss
from dll.optimizers import BaseOptimizer


class Model(BaseLayer):
    def __init__(self, layers: Sequence[BaseLayer]):
        super().__init__()
        self.layers: Sequence[BaseLayer] = layers

        # attributes to be set in compile()
        self.optimizer: Optional[BaseOptimizer] = None
        self.loss_function: Optional[_LossFunction] = None
        self.has_been_compiled = False

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

    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise AssertionError("Cannot call function `backward` on a Model instance.")

    def compile(self, loss_function: Type[_LossFunction], optimizer_class: Type[BaseOptimizer], **optimizer_args):
        self.loss_function = loss_function
        self.optimizer = optimizer_class(self, **optimizer_args)
        self.has_been_compiled = True

    def train(self,
              x: np.ndarray,
              y: np.ndarray,
              batch_size: int,
              epochs: int,
              validation_split: Optional[float] = 0.0,
              shuffle: Optional[bool] = True) -> None:

        assert self.has_been_compiled, "Must compile model before training"

        num_examples = x.shape[0]
        indices = np.array(range(num_examples))

        x_val, y_val = None, None

        if validation_split > 0:
            num_val_examples = int(num_examples * validation_split)
            np.random.shuffle(indices)
            x_val = x[indices[:num_val_examples]]
            y_val = y[indices[:num_val_examples]]

            x = x[indices[num_val_examples:]]
            y = y[indices[num_val_examples:]]

            num_examples = x.shape[0]
            indices = np.array(range(num_examples))

        num_batches = int(np.ceil(num_examples / batch_size))

        train_losses = []
        val_losses = []
        with tqdm(total=num_examples * epochs) as pbar:
            for epoch in range(epochs):
                if shuffle:
                    np.random.shuffle(indices)

                total_loss = 0

                for batch_idx in range(num_batches):
                    low = batch_idx * batch_size
                    high = min(num_examples, low + batch_size)
                    batch_indices = indices[low:high]
                    x_batch = x[batch_indices]
                    y_batch = y[batch_indices]

                    y_pred = self(x_batch)
                    loss, grad = self.loss_function.compute_loss(y_pred, y_batch)
                    self.optimizer.step(grad)

                    total_loss += loss

                    pbar.update(high - low)
                    pbar.set_description(f'Epoch {epoch + 1}, loss {total_loss / (batch_idx + 1): .3f}', refresh=True)

                train_losses.append(total_loss / num_batches)

                if validation_split > 0:
                    y_pred = self(x_val)
                    loss, _ = self.loss_function.compute_loss(y_pred, y_val)

                    val_losses.append(loss)

        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses, label='Val loss')
        plt.legend(loc='upper right')
        plt.show()

    def test(self, x: np.ndarray, y: np.ndarray) -> float:
        y_preds = self(x)
        if self.loss_function is CrossEntropyLoss:
            y_preds = np.argmax(y_preds, axis=-1)
            accuracy = np.mean(y_preds == y)
            return accuracy
        else:
            raise NotImplementedError()

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
