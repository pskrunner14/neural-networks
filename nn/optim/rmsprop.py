import numpy as np
np.random.seed(42)

from .optim import Optim

class RMSprop(Optim):

    def __init__(self, lr=.001, gamma=.9, epsilon=1e-7):
        super().__init__(lr)
        self._gamma = gamma
        self._epsilon = epsilon
        self._vars = None
        self._g2_vars = None

    @property
    def vars(self):
        return self._vars
        
    @vars.setter
    def vars(self, vars):
        self._g2_vars = []
        for var in vars:
            self._g2_vars.append(np.zeros_like(var.data))
        self._vars = vars

    def step(self):
        assert self._vars is not None, 'no trainable variables'
        assert self._g2_vars is not None, 'no optim specific params'
        # TODO (1): multithread this loop
        # can be optimized for faster computation
        for i in range(len(self._vars)):
            self._g2_vars[i] = (self._g2_vars[i] * self._gamma) + \
                               (np.square(self._vars[i].grad) * (1 - self._gamma))
            coeff = self._lr / (np.sqrt(self._g2_vars[i]) + self._epsilon)
            self._vars[i].update(coeff)