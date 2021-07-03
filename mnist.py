import os
import gzip
import numpy as np
import requests
import matplotlib.pyplot as plt

import dll
from dll import ReLU, CrossEntropyLoss
from dll.layers import Linear
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
        Linear(28 * 28, 128, ReLU),
        Linear(128, 10, None)
    ])
    model.print_summary()
    optimizer = SGD(model, learning_rate=0.01)

    batch_size = 128

    plt.ion()

    losses = []
    for epoch in range(0):
        sample = np.random.randint(0, x_train.shape[0], size=batch_size)
        x = x_train[sample].reshape((-1, 28 * 28))
        y = y_train[sample]

        y_pred = model(x)
        loss, grad = CrossEntropyLoss.compute_cost(y_pred, y)
        optimizer.step(grad)
        losses.append(loss)

        if epoch % 1000 == 0:
            print(f"{epoch = :>4}: {loss = :>8.3f}")
            plt.plot(losses)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

    y_preds_test = model(x_test.reshape((-1, 28 * 28)))
    y_preds_test = np.argmax(y_preds_test, axis=1)
    test_accuracy = np.mean(y_preds_test == y_test)

    print()
    print(f"Accuracy: {test_accuracy}")


if __name__ == '__main__':
    main()
