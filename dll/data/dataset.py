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
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x: np.ndarray = x
        self.y: np.ndarray = y

        assert self.x.shape[0] == self.y.shape[0], "x and y inputs must have the same length"

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class Subset(_BaseDataset):
    def __init__(self, dataset: _BaseDataset, indices: Sequence[int]):
        self.dataset: _BaseDataset = dataset
        self.indices: Sequence[int] = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


def random_split(dataset: _BaseDataset, lengths: Sequence[int]) -> List[Subset]:
    if len(dataset) != sum(lengths):
        raise ValueError("Length of dataset does not equal the sum of the input lengths")

    perm = np.random.permutation(len(dataset))

    subsets = []
    i = 0
    for l in lengths:
        subsets.append(Subset(dataset, perm[i: i + l]))
        i += l

    return subsets
