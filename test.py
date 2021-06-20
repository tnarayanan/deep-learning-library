import dll
from dll.layers import Linear

import numpy as np

model = dll.Model([
    Linear(10, 5),
    Linear(5, 3),
    Linear(3, 1)
])

x = np.ones((10, 1))
output = model(x)

print(output.shape)
print(output)
