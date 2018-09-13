from __future__ import print_function
import numpy as np

np.random.seed(42)

class Layer:
    
    """ A building block. Each layer is capable of performing two things:
        - Process input to get output:           output = layer.forward(input)
        - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)
        Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        pass
    
    def forward(self, input):
        """ Takes input data of shape [batch, input_units], 
            returns output data [batch, output_units]
        """
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input, grad_output, **kwargs):
        """ Performs a backpropagation step through the layer, with respect to the given input.
            To compute loss gradients w.r.t input, you need to apply chain rule (backprop):
            d loss / d x  = (d loss / d layer) * (d layer / d x)
            Luckily, you already receive d loss / d layer as input, 
            so you only need to multiply it by d layer / d x.
            If your layer has parameters (e.g. dense layer), 
            you also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input) # chain rule

class Dense(Layer):
    
    def __init__(self, input_units, output_units):
        """ A dense layer is a layer which performs a learned affine transformation:
            f(x) = <W*x> + b
        """
        # initialize weights with small random numbers. We use xavier initialization
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / (input_units + output_units))
        self.biases = np.zeros(output_units)
        self.g2_weights = np.zeros_like(self.weights)
        self.g2_biases = np.zeros_like(self.biases)
        
    def forward(self, input):
        """ Perform an affine transformation:
            f(x) = <W*x> + b
            
            input shape: [batch, input_units]
            output shape: [batch, output units]
        """
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, A_prev, dZ, optim='rmsprop', **kwargs):
        lr = kwargs['lr'] 
        alpha = kwargs['alpha']
        epsilon = kwargs['epsilon']

        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(dZ, self.weights.T)

        # dW = (1/m)*np.dot(dZ,np.transpose(A_prev))
        # db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        # dA_prev = np.dot(np.transpose(W),dZ)
        
        m = A_prev.shape[0]
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(A_prev.T, dZ) / m
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
    
class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        self.type = 'relu'
    
    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0, input)
    
    def backward(self, input, grad_output, **kwargs):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output * relu_grad