import numpy as np
np.random.seed(42)

from .layer import Layer
from nn.variable import Variable

class Dense(Layer):
    """ A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b

    Args:
        input_units (int): incoming connections to the dense layer.
        output_units (int): number of hidden neurons in the dense layer.
    """
    n = 0

    def __init__(self, input_units, output_units):
        self.type = 'layer'
        self._n = n

        # initialize weights with glorot/xavier uniform initialization
        self.W = Variable(data=np.random.randn(input_units, output_units) * np.sqrt(6. / (input_units + output_units)), name=f'W{n}')
        self.b = Variable(data=np.zeros(output_units), name=f'b{n}')
        self.trainable_vars = [self.W, self.b]
        n += 1

    def forward(self, x):
        """ Forward pass of the Dense Layer.
        Perform an affine transformation:
            f(x) = <W*x> + b

        input shape: [batch, input_units] 
        output shape: [batch, output units]

        Args:
            x (numpy.ndarray): the outputs from previous layers.
        Returns:
            numpy.ndarray: the linear transformation applied to the inputs.
        """
        self._x = x
        return np.dot(x, self.W.data) + self.b.data

    def backward(self, grads):
        """ Backward pass of the Dense Layer.
        Computes gradient of loss w.r.t. dense layer input.

        Args:
            grads (numpy.ndarray): the gradients w.r.t. loss propagated back from following layers.
        Returns:
            numpy.ndarray: the gradient of loss w.r.t. dense layer inputs.
        """
        # dL / dW = dL / dZ * dZ / dW = grads * x
        self.W.grad = np.dot(self._x.T, grads)
        # dL / db = dL / dZ * dZ / db = grads
        self.b.grad = np.mean(grads, axis=0)
        # dL / dx = dL / dZ * dZ / dx = grads * W.T
        # propagate back the gradients of Loss wrt to inputs x
        return np.dot(grads, self.W.data.T)