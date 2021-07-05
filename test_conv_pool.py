import numpy as np
from dll.layers import Conv2d, MaxPool2d, Linear

x = np.random.randn(3, 1, 6, 6)

conv = Conv2d(1, 8, 2)
conv_output = conv(x)
print(conv_output.shape)

conv_back, dw, _ = conv.backward(conv_output)
print(conv_back.shape)

pool = MaxPool2d(2)
pool_output = pool(x)
print(pool_output.shape)

pool_back, _, _ = pool.backward(pool_output)
print(pool_back.shape)