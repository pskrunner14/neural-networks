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
    
    def __init__(self, input_units, output_units, method='cuda_c'):
        self.method = method

        # initialize weights with small random numbers. We use xavier initialization
        self.weights = F.elemwise_prod(np.random.randn(input_units, output_units).astype('float32'), np.sqrt(2. / (input_units + output_units)), method=self.method)
        self.biases = np.zeros(output_units, dtype=np.float32)
        self.g2_weights = np.zeros_like(self.weights, dtype=np.float32)
        self.g2_biases = np.zeros_like(self.biases, dtype=np.float32)
        
    def forward(self, A):
        Wx = F.matmul(A, self.weights, method=self.method)
        Z = F.matsum(Wx, self.biases, method=self.method)
        return Z

    def backward(self, inputs, gradients, **kwargs):
        lr = kwargs['lr'] 
        alpha = kwargs['alpha']
        epsilon = kwargs['epsilon']
        optim = kwargs.get('optim', 'rmsprop')

        # dL / dx = dL / dZ * dZ / dx = gradients * W
        grad_input = F.matmul(gradients, self.weights.T, method=self.method)
        # m -> batch size
        m = inputs.shape[0]
        
        # compute gradient w.r.t. weights and biases
        # dL / dW = dL / dZ * dZ / dW = gradients * inputs
        grad_weights = F.elemwise_prod(F.matmul(inputs.T, gradients, method=self.method), 1 / m, method=self.method)
        # dL / db = dL / dZ * dZ / db = gradients * 1
        grad_biases = F.elemwise_prod(gradients.sum(axis=0), 1 / m, method=self.method)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        update_weights = F.elemwise_prod(grad_weights, lr, method=self.method)
        update_biases = F.elemwise_prod(grad_biases, lr, method=self.method)

        if optim == 'rmsprop':
            self.g2_weights = F.matsum(F.elemwise_prod(self.g2_weights, alpha, method=self.method), F.elemwise_prod(np.square(grad_weights), (1 - alpha), method=self.method), method=self.method)
            self.g2_biases = F.matsum(F.elemwise_prod(self.g2_biases, alpha, method=self.method), F.elemwise_prod(np.square(grad_biases), (1 - alpha), method=self.method), method=self.method)
            self.weights = F.matsum(self.weights, -F.matprod(update_weights, 1 / np.sqrt(F.elemwise_sum(self.g2_weights, epsilon, method=self.method)), method=self.method), method=self.method)
            self.biases = F.matsum(self.biases, -F.matprod(update_biases, 1 / np.sqrt(F.elemwise_sum(self.g2_biases, epsilon, method=self.method)), method=self.method), method=self.method)
        elif optim == 'gd':
            self.weights = F.matsum(self.weights, -update_weights, method=self.method)
            self.biases = F.matsum(self.biases, -update_biases, method=self.method)

        # propagate back the gradients of Loss wrt to layer inputs
        # dL / dx
        return grad_input
    
"""ReLU layer 

Applies elementwise rectified linear unit to all inputs:
    f(x) = max(0, x)

    input shape: [batch, input_units]
    output shape: [batch, input_units]
"""
class ReLU():

    def __init__(self, method='cuda_c'):
        self.type = 'relu'
        self.method = method
    
    def forward(self, A):
        return F.elemwise_max(A, 0, method=self.method)
    
    def backward(self, inputs, gradients, **kwargs):
        grad_relu = inputs > 0
        return F.matprod(gradients, grad_relu, method=self.method)