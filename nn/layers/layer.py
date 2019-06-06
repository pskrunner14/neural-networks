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
        raise NotImplementedError()

    def forward(self, inputs):
        """ Forward pass of the layer.
        Takes input data of shape [batch, input_units] 
        and returns output data [batch, output_units].
        """
        raise NotImplementedError()

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
        raise NotImplementedError()