import numpy as np

def softmax_1(z):
    return z / np.sum(z)

def softmax_2(z):
    z = np.e ** np.array(z)
    return z / np.sum(z)

z = [2.0, 1.0, 0.1]

print(softmax_1(z))
print(softmax_2(z))