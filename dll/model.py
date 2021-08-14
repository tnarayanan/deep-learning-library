from typing import Sequence, Optional, Type, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dll.data import DataLoader, _BaseDataset, random_split
from dll.layers import BaseLayer
from dll.loss_functions import _LossFunction, CrossEntropyLoss
from dll.optimizers import BaseOptimizer


class Model(BaseLayer):
    """A container for multiple neural network layers.

    The Model class provides a convenient way to train and evaluate neural networks.

    Args:
        layers: A sequence of individual layers that form the model.
    """

    def __init__(self, layers: Sequence[BaseLayer]):
        super().__init__()
        self.layers: Sequence[BaseLayer] = layers

        # attributes to be set in compile()
        self.output_shapes: List[Tuple[int, ...]] = []
        self.optimizer: Optional[BaseOptimizer] = None
        self.loss_function: Optional[Type[_LossFunction]] = None
        self.has_been_compiled = False

    def get_num_params(self) -> int:
        total_params = 0
        for layer in self.layers:
            total_params += layer.get_num_params()
        return total_params

    def forward(self, x: np.ndarray, is_training: Optional[bool] = True) -> np.ndarray:
        """Executes a forward-pass of the model with the given input.

        Args:
            x: A Numpy array representing the input to the model.
            is_training: A boolean representing whether the forward pass occurs during model training or evaluation.

        Returns:
            The output of the forward-pass of the model.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise AssertionError("Cannot call function `backward` on a Model instance.")

    def compile(self, input_shape: Tuple[int, ...], loss_function: Type[_LossFunction],
                optimizer_class: Type[BaseOptimizer], **optimizer_args):
        """Compiles the model to prepare for training.

        Args:
            input_shape: The shape of the input to the model.
            loss_function: The loss function to be used after the forward pass of the model.
            optimizer_class: The optimizer class to be used to optimize the model.
            optimizer_args: Keyword arguments to be passed into the optimizer's constructor.

        Raises:
            ValueError: The layer input/output dimensions do not match.
        """
        # run empty data through model to check for layer dimension mismatches
        self.output_shapes.clear()
        self.output_shapes.append(input_shape)
        x = np.zeros((1,) + input_shape)
        for i, layer in enumerate(self.layers):
            try:
                x = layer(x)
                self.output_shapes.append(x.shape[1:])
            except ValueError as e:
                raise ValueError(f"Layer dimensions do not match") from e

        self.loss_function = loss_function
        self.optimizer = optimizer_class(self, **optimizer_args)
        self.has_been_compiled = True

    def train(self,
              train_dataset: _BaseDataset,
              batch_size: int,
              epochs: int,
              val_split: float = 0.0,
              val_dataset: Optional[_BaseDataset] = None,
              shuffle: Optional[bool] = True) -> None:
        """Train the model on a dataset.

        Args:
            train_dataset: The dataset used to train the model.
            batch_size: The size of each batch during training.
            epochs: The number of epochs to train the model for.
            val_split: Optional; how much of the train dataset to use for validation. This value is
              ignored if `val_dataset` is specified.
            val_dataset: Optional; the dataset used for validating the model during training.
            shuffle: Optional; a boolean representing whether to shuffle the batches of data every epoch.
        """
        assert self.has_been_compiled, "Must compile model before training"
        assert self.loss_function is not None
        assert self.optimizer is not None

        if val_split > 0 and val_dataset is None:
            len_val_dataset = int(val_split * len(train_dataset))
            train_dataset, val_dataset = random_split(train_dataset,
                                                      [len(train_dataset) - len_val_dataset, len_val_dataset])

        train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle)

        num_examples = len(train_dataset)
        num_batches = int(np.ceil(num_examples / batch_size))

        train_losses = []
        val_losses = []
        with tqdm(total=num_examples * epochs) as pbar:
            for epoch in range(epochs):
                total_loss = 0

                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    y_pred = self(x_batch)
                    loss, grad = self.loss_function.compute_loss(y_pred, y_batch)
                    self.optimizer.step(grad)

                    total_loss += loss
                    pbar.update(len(y_batch))
                    pbar.set_description(f'Epoch {epoch + 1}, loss {total_loss / (batch_idx + 1): .3f}', refresh=True)

                train_losses.append(total_loss / num_batches)

                if val_dataset is not None:
                    x_val, y_val = val_dataset[:]
                    y_pred = self(x_val)
                    loss, _ = self.loss_function.compute_loss(y_pred, y_val)
                    val_losses.append(loss)

        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses, label='Val loss')
        plt.legend(loc='upper right')
        plt.show()

    def test(self, test_dataset: _BaseDataset) -> float:
        """Train the model on a dataset.

        Args:
            train_dataset: The dataset used to train the model.
            batch_size: The size of each batch during training.
            epochs: The number of epochs to train the model for.
            val_split: Optional; how much of the train dataset to use for validation. This value is
              ignored if `val_dataset` is specified.
            val_dataset: Optional; the dataset used for validating the model during training.
            shuffle: Optional; a boolean representing whether to shuffle the batches of data every epoch.
        """
        x, y = test_dataset[:]
        y_preds = self(x)
        if self.loss_function is CrossEntropyLoss:
            y_preds = np.argmax(y_preds, axis=-1)
            accuracy = np.mean(y_preds == y)
            return accuracy[()]
        else:
            raise NotImplementedError()

    def print_summary(self) -> None:
        """Prints a summary of the model.

        The summary includes details about each layer:
          * Layer type
          * Output shape
          * Number of parameters

        The summary also includes the total number of parameters in the model.
        """
        assert self.has_been_compiled, "Must compile model before printing summary"

        left_space_len = 2
        layer_padding = 12
        output_shape_padding = 20
        params_padding = 8
        right_space_len = 2

        left_space = " " * left_space_len
        right_space = " " * right_space_len

        def print_line(layer_str: str, shape: str, params: str):
            print(
                f'{left_space}{layer_str: >{layer_padding - 3}}   {shape: <{output_shape_padding - 2}}  {params: >{params_padding}}{right_space}')

        thick_line = "=" * (left_space_len + layer_padding + output_shape_padding + params_padding + right_space_len)
        thin_line = "—" * (left_space_len + layer_padding + output_shape_padding + params_padding + right_space_len)

        print_line("Layer", "Output Shape", "Num Params")
        print(thick_line)
        print_line("Input", repr((np.newaxis,) + self.output_shapes[0]), "")

        for i, layer in enumerate(self.layers):
            print(thin_line)
            print_line(layer.get_name(), repr((np.newaxis,) + self.output_shapes[i + 1]), str(layer.get_num_params()))

        print(thick_line)
        print(f'{left_space}Total params: {self.get_num_params()}')
        print()
