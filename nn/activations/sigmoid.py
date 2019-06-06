import numpy as np
np.random.seed(42)

from nn.layers import Layer

class Sigmoid(Layer):

    def __init__(self):
        self.type = 'activation'

    def forward(self, X):
        return 1. / 1. + (np.exp(-1. * X))

    def backward(self):
        pass