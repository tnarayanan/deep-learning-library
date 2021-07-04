import dll
from dll.layers import Linear, Flatten
import dll.data as data
import dll.optimizers as optimizers


def main():
    train_dataset = data.MNISTDataset(train=True)
    test_dataset = data.MNISTDataset(train=False)

    model = dll.Model([
        Flatten((28, 28)),
        Linear(28 * 28, 128, dll.ReLU),
        Linear(128, 10)
    ])
    model.print_summary()

    model.compile(dll.CrossEntropyLoss, optimizers.SGD, learning_rate=0.01)
    model.train(train_dataset, batch_size=128, epochs=50, val_split=0.1)

    print()
    print(f"Accuracy: {model.test(test_dataset): .3%}")


if __name__ == '__main__':
    main()
