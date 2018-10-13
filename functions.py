import numpy as np

from ops.cpu_ops import (
    cpu_matmul,
    cpu_matsum,
    cpu_matprod,
    cpu_elemwise_max
)
from ops.numba_ops import (
    numba_matmul,
    numba_matsum,
    numba_matprod,
    numba_elemwise_max
)
from ops.cuda_c_ops import (
    cuda_matmul,
    cuda_matsum,
    cuda_matprod
)

""" Computation functions.

Offload computation efficiently on respective device.
"""
def compute_matmul(a, b, method='cpu'):
    if method == 'cpu':
        pass
    elif method == 'cuda_c':
        m, n, k = a.shape[0], a.shape[1], b.shape[1]
        a = a.flatten()
        b = b.flatten()
        c = np.zeros(shape=(m * k), dtype=np.float32)
        cuda_matmul(a.astype(np.float32), b.astype(np.float32), c, m, n, k)
        return c.reshape((m, k))
    elif method == 'numba':
        pass
    else:
        raise UserWarning('Unknown computation method.')

def compute_matsum(a, b, method='cpu'):
    if method == 'cpu':
        pass
    elif method == 'cuda_c':
        m, n = a.shape[0], a.shape[1]
        a = a.flatten()
        b = b.flatten()
        c = np.zeros_like(a=a, dtype=np.float32)
        cuda_matsum(a.astype(np.float32), b.astype(np.float32), c, m, n)
        return c.reshape((m, n))
    elif method == 'numba':
        pass
    else:
        raise UserWarning('Unknown computation method.')

def compute_matprod(a, b, method='cpu'):
    if method == 'cpu':
        pass
    elif method == 'cuda_c':
        pass
    elif method == 'numba':
        pass
    else:
        raise UserWarning('Unknown computation method.')

def compute_elemwise_max(a, value, method='cpu'):
    if method == 'cpu':
        pass
    elif method == 'cuda_c':
        pass
    elif method == 'numba':
        pass
    else:
        raise UserWarning('Unknown computation method.')