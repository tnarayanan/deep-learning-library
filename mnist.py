import numpy as np

import dll
from dll.layers import Linear, Flatten, Conv2d, MaxPool2d
import dll.data as data
import dll.optimizers as optimizers


def main():
    train_dataset = data.MNISTDataset(train=True)
    test_dataset = data.MNISTDataset(train=False)

    # model = dll.Model([
    #     Flatten((1, 28, 28)),
    #     Linear(28 * 28, 128, dll.ReLU),
    #     Linear(128, 10)
    # ])
    model = dll.Model([
        Conv2d(1, 32, 3, activation=dll.ReLU),
        MaxPool2d(2),
        Flatten((32, 13, 13)),
        Linear(32 * 13 * 13, 100, activation=dll.ReLU),
        Linear(100, 10)
    ])

    model.compile((1, 28, 28), dll.CrossEntropyLoss, optimizers.SGD, learning_rate=0.01)
    model.print_summary()
    model.train(train_dataset, batch_size=32, epochs=5, val_split=0.1)

    print()
    print(f"Accuracy: {model.test(test_dataset): .3%}")


if __name__ == '__main__':
    main()
