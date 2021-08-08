# DLL: Deep Learning Library

A deep learning library written from scratch using Numpy.

DLL is not GPU accelerated and does not contain nearly as many features as other
deep learning libraries like PyTorch or Keras, so it's not practical to use. I made
DLL to better understand the math behind deep learning, and because it was fun!

## Example

```python
import dll
from dll.layers import Linear, Flatten
import dll.data as data
import dll.optimizers as optimizers

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

Epoch 50, loss  0.062: 100%|██████████| 2700000/2700000 [01:20<00:00, 33445.49it/s]

Accuracy: 97.460%
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
