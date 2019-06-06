import numpy as np
np.random.seed(42)

class Optim():

    def __init__(self, lr):
        self.lr = lr

    def step(self):
        raise NotImplementedError()