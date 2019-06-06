import numpy as np
np.random.seed(42)

from .optim import Optim

class SGD(Optim):

    def __init__(self, *args, **kwargs):
        pass
    
    def step():
        lr = kwargs.get('lr', 0.001)

        # dL / dx = dL / dZ * dZ / dx = gradients * W
        grad_input = np.dot(gradients, self.weights.T)
        # m -> batch size
        m = inputs.shape[0]

        # compute gradient w.r.t. weights and biases
        # dL / dW = dL / dZ * dZ / dW = gradients * inputs
        grad_weights = np.dot(inputs.T, gradients) / m
        # dL / db = dL / dZ * dZ / db = gradients * 1
        grad_biases = gradients.sum(axis=0) / m

        assert grad_weights.shape == self.weights.shape and \
            grad_biases.shape == self.biases.shape

        update_weights = lr * grad_weights
        update_biases = lr * grad_biases

        self.weights -= update_weights
        self.biases -= update_biases