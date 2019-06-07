import numpy as np
np.random.seed(42)

class Variable:
    # keeps count of unnamed variables 
    # so there are only uniques on the graph
    n = 0

    def __init__(self, data: np.array, name: str='x', requires_grad: bool=True):
        if name == 'x':
            self.name = f'x{n}'
            n += 1
        else:
            self.name = name
        self._data = data
        self._grad = None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f'Variable: {self.name} {self._data.shape}' + \
                '[requires_grad]' if self.requires_grad else ''

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.array):
        self._data = data

    @data.deleter
    def data(self):
        self._data = None

    @property
    def grad(self):
        assert self.requires_grad, "variable doesn't support differentiation"
        return self._grad

    @grad.setter
    def grad(self, grad: np.array):
        assert self.requires_grad, "variable doesn't support differentiation"
        assert grad.shape == self._data.shape, \
               f'gradient shape mismatch: {grad.shape} -> {self._data.shape}'
        self._grad = grad
        
    @grad.deleter
    def grad(self):
        assert self.requires_grad, "variable doesn't support differentiation"
        self._grad = None

    def update(self, coeff):
        self.data -= coeff * self.grad
        del self.grad