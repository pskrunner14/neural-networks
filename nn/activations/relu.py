import numpy as np
np.random.seed(42)

from nn.layers import Layer

class ReLU(Layer):
    """ Simply applies elementwise rectified linear unit to all inputs. """

    def __init__(self):
        self.type = 'activation'

    def forward(self, inputs):
        """ Forward pass of the ReLU non-linearity.

        Args:
            inputs (numpy.ndarray): the linear transformation from dense layer.
        """
        return np.maximum(0, inputs)

    def backward(self, inputs, gradients, **kwargs):
        """ Backward pass of the ReLU non-linearity.
        Computes gradient of loss w.r.t. ReLU input.

        Args:
            inputs (numpy.ndarray): the inputs to the ReLU layer to compute the gradients.
            gradients (numpy.ndarray): the gradients w.r.t. loss propagated back from following layers.
        Returns:
            numpy.ndarray: the gradient of loss w.r.t. ReLU inputs.
        """
        grad_relu = inputs > 0
        return gradients * grad_relu