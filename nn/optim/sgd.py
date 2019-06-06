import numpy as np
np.random.seed(42)

from .optim import Optim

class SGD(Optim):

    def __init__(self, lr=.001):
        super().__init__(lr)
        self._vars = None

    @property
    def vars(self):
        return self._vars
        
    @vars.setter
    def vars(self, vars):
        self._vars = vars

    def step(self):
        assert self._vars is not None, 'no trainable variables'
        for i in range(len(self._vars)):
            self._vars[i].data -= self._lr * self._vars[i].grad