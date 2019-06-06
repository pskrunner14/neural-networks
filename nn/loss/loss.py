import numpy as np
np.random.seed(42)

class Loss():

    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError