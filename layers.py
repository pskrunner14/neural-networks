from __future__ import print_function
import numpy as np

import functions as F

np.random.seed(42)

""" Dense Layer

Performs a learned affine transformation:
    f(x) = <W*x> + b

    input shape: [batch, input_units]
    output shape: [batch, output units]
"""
class Dense():
    
    def __init__(self, input_units, output_units, method='cpu'):
        # initialize weights with small random numbers. We use xavier initialization
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / (input_units + output_units))
        self.biases = np.zeros(output_units)
        self.g2_weights = np.zeros_like(self.weights)
        self.g2_biases = np.zeros_like(self.biases)

        self.method = method
        
    def forward(self, A):
        Wx = F.matmul(A, self.weights, method=self.method)
        bias = self.biases
        bias = np.repeat(bias, Wx.shape[0], axis=0)
        Z = F.matsum(Wx, bias, method=self.method)
        return Z

    def backward(self, A_prev, dZ, **kwargs):
        lr = kwargs['lr'] 
        alpha = kwargs['alpha']
        epsilon = kwargs['epsilon']
        optim = kwargs.get('optim', 'rmsprop')

        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = F.matmul(dZ, self.weights.T, method=self.method)
        m = A_prev.shape[0]
        
        # compute gradient w.r.t. weights and biases
        grad_weights = F.matmul(A_prev.T, dZ, method=self.method) / m
        grad_biases = dZ.sum(axis=0) / m
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        update_weights = lr * grad_weights
        update_biases = lr * grad_biases

        if optim == 'rmsprop':
            self.g2_weights = (alpha * self.g2_weights) + (1 - alpha) * np.square(grad_weights)
            self.g2_biases = (alpha * self.g2_biases) + (1 - alpha) * np.square(grad_biases)
            # Here we perform a stochastic gradient descent step. 
            # Later on, you can try replacing that with something better.
            self.weights -= update_weights / np.sqrt(self.g2_weights + epsilon)
            self.biases -= update_biases / np.sqrt(self.g2_biases + epsilon)
        elif optim == 'gd':
            self.weights -= update_weights
            self.biases -= update_biases

        return grad_input
    
"""ReLU layer 

Applies elementwise rectified linear unit to all inputs:
    f(x) = max(0, x)

    input shape: [batch, input_units]
    output shape: [batch, input_units]
"""
class ReLU():

    def __init__(self, method='cpu'):
        self.type = 'relu'
        self.method = method
    
    def forward(self, A):
        return np.maximum(0, A)
    
    def backward(self, A_prev, dZ, **kwargs):
        relu_grad = A_prev > 0
        return dZ * relu_grad