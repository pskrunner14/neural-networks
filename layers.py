from __future__ import print_function
import numpy as np

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

class Dense(Layer):
    """ Dense layer.
    A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
    """
    
    def __init__(self, input_units, output_units):
        # initialize weights with small random numbers. We use xavier initialization
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / (input_units + output_units))
        self.biases = np.zeros(output_units)
        self.g2_weights = np.zeros_like(self.weights)
        self.g2_biases = np.zeros_like(self.biases)
        
    def forward(self, inputs):
        """ Forward pass of the dense layer.
        Perform an affine transformation:
            f(x) = <W*x> + b
            
        input shape: [batch, input_units] 
        output shape: [batch, output units]
        """
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, inputs, gradients, **kwargs):
        lr = kwargs.get('lr', 0.001)
        alpha = kwargs.get('alpha', 0.99)
        epsilon = kwargs.get('epsilon', 1e-8)
        optim = kwargs.get('optim', 'rmsprop')

        # dL / dx = dL / dZ * dZ / dx = gradients * W
        grad_input = np.dot(gradients, self.weights.T)
        # m -> batch size
        m = inputs.shape[0]
        
        # compute gradient w.r.t. weights and biases
        # dL / dW = dL / dZ * dZ / dW = gradients * inputs
        grad_weights = np.dot(inputs.T, gradients) / m
        # dL / db = dL / dZ * dZ / db = gradients * 1
        grad_biases = gradients.sum(axis=0) / m
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        update_weights = lr * grad_weights
        update_biases = lr * grad_biases

        if optim == 'rmsprop':
            self.g2_weights = (alpha * self.g2_weights) + (1 - alpha) * np.square(grad_weights)
            self.g2_biases = (alpha * self.g2_biases) + (1 - alpha) * np.square(grad_biases)
            
            self.weights -= update_weights / np.sqrt(self.g2_weights + epsilon)
            self.biases -= update_biases / np.sqrt(self.g2_biases + epsilon)
        elif optim == 'gd':
            self.weights -= update_weights
            self.biases -= update_biases

        # propagate back the gradients of Loss wrt to layer inputs
        # dL / dx
        return grad_input
    
class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        self.type = 'relu'
    
    def forward(self, inputs):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0, inputs)
    
    def backward(self, inputs, gradients, **kwargs):
        """Compute gradient of loss w.r.t. ReLU input"""
        grad_relu = inputs > 0
        return gradients * grad_relu