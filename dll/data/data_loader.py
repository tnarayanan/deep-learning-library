from typing import Optional

import numpy as np

from dll.data import _BaseDataset


class DataLoader(object):
    def __init__(self, dataset: _BaseDataset, batch_size: int, shuffle: Optional[bool] = True):
        self.dataset: _BaseDataset = dataset
        self.num_examples: int = len(self.dataset)

        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle

        self._indices: np.ndarray = np.array(range(self.num_examples))
        self._i: int = 0

    def shuffle_indices(self):
        np.random.shuffle(self._indices)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        self._i = 0
        return self

    def __next__(self):
        if self._i == self.num_examples:
            raise StopIteration

        high = min(self.num_examples, self._i + self.batch_size)
        batch = self.dataset[self._indices[self._i:high]]
        self._i = high

        return batch
