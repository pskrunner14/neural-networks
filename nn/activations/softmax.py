import numpy as np
np.random.seed(42)

from nn.layers import Layer

class Softmax(Layer):

    def __init__(self):
        self.type = 'activation'

    def forward(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def backward(self):
        pass