import os
import gzip
import numpy as np
import requests
import matplotlib.pyplot as plt

import dll
from dll.layers import Linear, Flatten
from dll import ReLU, CrossEntropyLoss
from dll.optimizers import SGD


def download_mnist_dataset():
    d = []
    os.makedirs("datasets/mnist", exist_ok=True)
    for p, url in zip(['train_images.gz', 'train_labels.gz', 'test_images.gz', 'test_labels.gz'],
                      ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                       'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                       'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                       'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']):
        fp = os.path.join('datasets', 'mnist', p)
        if os.path.isfile(fp):
            with open(fp, 'rb') as f:
                data = f.read()
        else:
            with open(fp, 'wb') as f:
                data = requests.get(url).content
                f.write(data)

        d.append(np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy())

    return d


def main():
    x_train, y_train, x_test, y_test = download_mnist_dataset()
    x_train = x_train[16:].reshape((-1, 28, 28)) / 255
    y_train = y_train[8:]
    x_test = x_test[16:].reshape((-1, 28, 28)) / 255
    y_test = y_test[8:]

    model = dll.Model([
        Flatten((28, 28)),
        Linear(28 * 28, 128, ReLU),
        Linear(128, 10)
    ])
    model.print_summary()

    model.compile(CrossEntropyLoss, SGD, learning_rate=0.01)
    model.train(x_train, y_train, batch_size=128, epochs=2, validation_split=0.1)

    print()
    print(f"Accuracy: {model.test(x_test, y_test): .3%}")



if __name__ == '__main__':
    main()
