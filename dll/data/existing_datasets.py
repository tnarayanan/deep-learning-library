import gzip
import os
from typing import Tuple

import numpy as np
import requests

from dll.data import Dataset


class MNISTDataset(Dataset):
    """The MNIST dataset of handwritten digits.

    Args:
        train: A boolean representing whether to load train data (True) or test data (False).
        save_path: A string representing the folder location of the MNIST data.
        download: A boolean representing whether to download the data to the save_path.

    Raises:
        RuntimeError: The MNIST data was not found in the save_path.
    """
    train_sources: Tuple[str, str] = [
        ('train_images.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'),
        ('train_labels.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')]
    test_sources: Tuple[str, str] = [
        ('test_images.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'),
        ('test_labels.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')]

    def __init__(self, train: bool = True, save_path: str = 'datasets/mnist', download: bool = False):
        d = []
        os.makedirs(save_path, exist_ok=True)

        if download:
            MNISTDataset.download_data(save_path)

        if train:
            source = MNISTDataset.train_sources
        else:
            source = MNISTDataset.test_sources

        for p, url in source:
            fp = os.path.join('datasets', 'mnist', p)
            if os.path.isfile(fp):
                with open(fp, 'rb') as f:
                    data = f.read()
            else:
                raise RuntimeError(f'MNIST data not found in {save_path}. Use download=True to download the dataset.')

            d.append(np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy())

        d[0] = d[0][16:].reshape((-1, 1, 28, 28)) / 255
        d[1] = d[1][8:]

        super().__init__(d[0], d[1])

    @staticmethod
    def download_data(save_path):
        for source in (MNISTDataset.train_sources, MNISTDataset.test_sources):
            for p, url in source:
                fp = os.path.join(save_path, p)
                if not os.path.isfile(fp):
                    with open(fp, 'wb') as f:
                        data = requests.get(url).content
                        f.write(data)
