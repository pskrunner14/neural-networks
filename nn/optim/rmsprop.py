import numpy as np
np.random.seed(42)

from .optim import Optim

class RMSprop():
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

    gamma = kwargs.get('gamma', 0.9)
    epsilon = kwargs.get('epsilon', 1e-7)
    if not hasattr(self, 'g2_weights'):
        self._init_g2()
    self.g2_weights = (self.g2_weights * gamma) + \
        np.square(grad_weights) * (1 - gamma)
    self.g2_biases = (self.g2_biases * gamma) + \
        np.square(grad_biases) * (1 - gamma)

    self.weights -= update_weights / \
        (np.sqrt(self.g2_weights) + epsilon)
    self.biases -= update_biases / (np.sqrt(self.g2_biases) + epsilon)