# DLL: Deep Learning Library

A deep learning library written from scratch using Numpy.

DLL is not GPU accelerated and does not contain nearly as many features as other
deep learning libraries like PyTorch or Keras, so it's not practical to use. I made
DLL to better understand the math behind deep learning, and because it was fun!

## Example

```python
import dll
from dll.layers import Linear, Flatten
import dll.optimizers as optimizers

x_train, y_train, x_test, y_test = get_mnist_data()

model = dll.Model([
        Flatten((28, 28), 28 * 28),
        Linear(28 * 28, 128, dll.ReLU),
        Linear(128, 10, None)
])
model.print_summary()

model.compile(dll.CrossEntropyLoss, optimizers.SGD, learning_rate=0.01)
model.train(x_train, y_train, batch_size=128, epochs=50, validation_split=0.1)

accuracy = model.test(x_test, y_test)
print(f"Accuracy: {accuracy: .3%}")
```

**Output**:

```
      Layer   Output Shape   # Params  
=======================================
      Input   (None, 28, 28)            
———————————————————————————————————————
    Flatten   (None, 784)           0  
———————————————————————————————————————
     Linear   (None, 128)      100480  
———————————————————————————————————————
     Linear   (None, 10)         1290  
=======================================
  Total params: 101770

Epoch 50, loss  0.193: 100%|██████████| 2700000/2700000 [00:34<00:00, 77146.34it/s]

Accuracy: 94.420%
```

## Features

* Layers
    * Linear
    * Flatten
* Activations
    * Sigmoid
    * ReLU
    * Tanh
* Loss functions
    * Binary cross-entropy (BCE)
    * Cross-entropy
    * Mean-squared error (MSE)
* Optimizers
    * Stochastic gradient descent (SGD)
