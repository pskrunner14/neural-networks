from __future__ import print_function
import numpy as np

import functions as F

np.random.seed(42)

class Layer:
    """ Implements a layer module.
    A building block. Each layer is capable of performing two things:
        - Process input to get output: output = layer.forward(inputs)
        - Propagate gradients through itself: grad_input = layer.backward(inputs, gradients)
    Some layers also have learnable parameters which they update during the backward pass.
    """

    def __init__(self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        pass
    
    def forward(self, inputs):
        """ Forward pass of the layer.
        Takes input data of shape [batch, input_units], 
        and returns output data [batch, output_units]
        """
        pass

    def backward(self, inputs, gradients, **kwargs):
        """ Backward pass of the layer.
        Performs a backpropagation step through the layer, with respect to the given input.
        To compute loss gradients w.r.t input, you need to apply chain rule (backprop):
            dL / dx  = (dL / dZ) * (dZ / dx)
            Luckily, we already receive dL / dZ as input, 
            so we only need to multiply it by dZ / dx.
            If our layer has parameters (e.g. dense layer, conv layer etc.), 
            you also need to update them here using dL / dZ.
        """
        pass

class Dense():
    """ Dense layer.
    A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
    input shape: [batch, input_units]
    output shape: [batch, output units]
    """

    def __init__(self, input_units, output_units, method='cpu'):
        self.type = 'dense'
        self.method = method

        # initialize weights with small random numbers. We use xavier initialization
        self.weights = F.prod(np.random.randn(input_units, output_units), np.sqrt(2. / (input_units + output_units)), method=self.method)
        self.biases = np.zeros(output_units)

    def _init_g2(self):
        self.g2_weights = np.zeros_like(self.weights)
        self.g2_biases = np.zeros_like(self.biases)

    def forward(self, inputs):
        """ Forward pass of the dense layer.
        Perform an affine transformation:
            f(x) = <W*x> + b
            
        input shape: [batch, input_units] 
        output shape: [batch, output units]
        """
        Wx = F.matmul(inputs, self.weights, method=self.method)
        Z = F.matsum(Wx, self.biases, method=self.method)
        return Z

    def backward(self, inputs, gradients, **kwargs):
        """ Backward pass of the layer.
        Performs a backpropagation step through the layer, with respect to the given input.
        To compute loss gradients w.r.t input, you need to apply chain rule (backprop):
            dL / dx  = (dL / dZ) * (dZ / dx)
        """
        lr = kwargs.get('lr', 0.001)
        gamma = kwargs.get('gamma', 0.9)
        epsilon = kwargs.get('epsilon', 1e-7)
        optim = kwargs.get('optim', 'rmsprop')

        # dL / dx = dL / dZ * dZ / dx = gradients * W
        grad_input = F.matmul(gradients, self.weights.T, method=self.method)
        # m -> batch size
        m = inputs.shape[0]
        
        # compute gradient w.r.t. weights and biases
        # dL / dW = dL / dZ * dZ / dW = gradients * inputs
        grad_weights = F.prod(F.matmul(inputs.T, gradients, method=self.method), 1. / m, method=self.method)
        # dL / db = dL / dZ * dZ / db = gradients * 1
        grad_biases = F.prod(gradients.sum(axis=0), 1. / m, method=self.method)
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        update_weights = F.prod(grad_weights, lr, method=self.method)
        update_biases = F.prod(grad_biases, lr, method=self.method)

        if optim == 'rmsprop':
            if not hasattr(self, 'g2_weights'):
                self._init_g2()
            self.g2_weights = F.matsum(F.prod(self.g2_weights, gamma, method=self.method), F.prod(np.square(grad_weights), (1 - gamma), method=self.method), method=self.method)
            self.g2_biases = F.matsum(F.prod(self.g2_biases, gamma, method=self.method), F.prod(np.square(grad_biases), (1 - gamma), method=self.method), method=self.method)
            self.weights = F.matsum(self.weights, -F.matprod(update_weights, 1. / np.sqrt(F.add(self.g2_weights, epsilon, method=self.method)), method=self.method), method=self.method)
            self.biases = F.matsum(self.biases, -F.matprod(update_biases, 1. / np.sqrt(F.add(self.g2_biases, epsilon, method=self.method)), method=self.method), method=self.method)
        elif optim == 'gd':
            self.weights = F.matsum(self.weights, -update_weights, method=self.method)
            self.biases = F.matsum(self.biases, -update_biases, method=self.method)

        # propagate back the gradients of Loss wrt to layer inputs
        # dL / dx
        return grad_input

class ReLU():
    """ ReLU layer.

    Applies elementwise rectified linear unit to all inputs:
        f(x) = max(0, x)

        input shape: [batch, input_units]
        output shape: [batch, input_units]
    """

    def __init__(self, method='cpu'):
        self.type = 'relu'
        self.method = method
    
    def forward(self, inputs):
        return F.maximum(inputs, 0., method=self.method)
    
    def backward(self, inputs, gradients, **kwargs):
        grad_relu = inputs > 0.
        return F.matprod(gradients, grad_relu, method=self.method)
