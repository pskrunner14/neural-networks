import numpy as np
np.random.seed(42)

from .layer import Layer

class Dense(Layer):
    """ A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b

    Args:
        input_units (int): incoming connections to the dense layer.
        output_units (int): number of hidden neurons in the dense layer.
    """

    def __init__(self, input_units, output_units):
        self.type = 'layer'

        # initialize weights with glorot/xavier uniform initialization
        self.weights = np.random.randn(input_units, output_units) * \
            np.sqrt(6. / (input_units + output_units))
        self.biases = np.zeros(output_units)

    def _init_g2(self):
        self.g2_weights = np.zeros_like(self.weights)
        self.g2_biases = np.zeros_like(self.biases)

    def forward(self, inputs):
        """ Forward pass of the Dense Layer.
        Perform an affine transformation:
            f(x) = <W*x> + b

        input shape: [batch, input_units] 
        output shape: [batch, output units]

        Args:
            inputs (numpy.ndarray): the outputs from previous layers.
        Returns:
            numpy.ndarray: the linear transformation applied to the inputs.
        """
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, inputs, gradients, **kwargs):
        """ Backward pass of the Dense Layer.
        Computes gradient of loss w.r.t. dense layer input.

        Args:
            inputs (numpy.ndarray): the inputs to the dense layer to compute the gradients.
            gradients (numpy.ndarray): the gradients w.r.t. loss propagated back from following layers.
        Returns:
            numpy.ndarray: the gradient of loss w.r.t. dense layer inputs.
        """
        # dL / dx = dL / dZ * dZ / dx = gradients * W
        grad_input = np.dot(gradients, self.weights.T)

        # propagate back the gradients of Loss wrt to layer inputs
        # dL / dx
        return grad_input