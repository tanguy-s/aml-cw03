import numpy as np


a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]

a = np.array(a).reshape([-1, 1])
b = np.array(b).reshape([-1, 1])

print(a)
print(b)
print(np.concatenate([a, b], axis=1))


