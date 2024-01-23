import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/ (1 + np.e ** -z)

for z in np.linspace(-10, 10, 20):
    s = sigmoid(z)
    print('{:.5} : {:.5}'.format(z, s))

    plt.plot(z, s, 'ro')
plt.show()

def cross_entropy(y):
    def log_a():
        return 'a'
    
    def log_b():
        return 'b'
    
    print(y * log_a() + (1-y) * log_b())

cross_entropy(y=0)
cross_entropy(y=1)