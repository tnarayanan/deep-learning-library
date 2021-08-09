from abc import ABC, abstractmethod
from typing import Sequence, List

import numpy as np


class _BaseDataset(ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError()


class Dataset(_BaseDataset):
    """A container for a set of data.

    Datasets have inputs (x) and outputs (y). Each example has a corresponding output.

    Args:
        x: A Numpy array of the inputs.
        y: A Numpy array of the outputs.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x: np.ndarray = x
        self.y: np.ndarray = y

        assert self.x.shape[0] == self.y.shape[0], "x and y inputs must have the same length"

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class Subset(_BaseDataset):
    """A container representing a subset of a dataset.

    Subset is a more efficient way of representing a portion of an existing dataset, as it does not copy the
    underlying data. Instead, it only stores the indices of the examples in the existing dataset.

    Args:
        dataset: The original dataset.
        indices: A sequence of integers that represents the indices of the original dataset that should be
          included in the Subset.
    """

    def __init__(self, dataset: _BaseDataset, indices: Sequence[int]):
        self.dataset: _BaseDataset = dataset
        self.indices: Sequence[int] = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


def random_split(dataset: _BaseDataset, subset_lengths: Sequence[int]) -> List[Subset]:
    """Randomly splits a dataset into multiple subsets of specified lengths.

    Args:
        dataset: The dataset to be split into Subsets.
        subset_lengths: A sequence of integers representing the lengths of each subset.

    Returns:
        A list of Subsets of the original dataset. These subsets are in the same order as the inputted subset_lengths.

    Raises:
        ValueError: The sum of the subset lengths does not equal the length of the dataset.
    """
    if len(dataset) != sum(subset_lengths):
        raise ValueError("The sum of the input lengths does not equal the length of the dataset")

    perm = np.random.permutation(len(dataset))

    subsets = []
    i = 0
    for length in subset_lengths:
        subsets.append(Subset(dataset, perm[i: i + length]))
        i += length

    return subsets
