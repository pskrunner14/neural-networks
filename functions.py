import numpy as np

from ops.cpu_ops import (
    cpu_matmul,
    cpu_matsum,
    cpu_matprod,
    cpu_elemwise_sum,
    cpu_elemwise_prod,
    cpu_elemwise_max
)
from ops.numba_ops import (
    numba_matmul,
    numba_matsum,
    numba_matprod,
    numba_elemwise_sum,
    numba_elemwise_prod,
    numba_elemwise_max
)

NUM_THREADS = 32

def get_cuda_execution_config(m, n):
    gridBlock = (NUM_THREADS, NUM_THREADS)
    gridDim = ((n // gridBlock[0]) + 1, (m // gridBlock[1]) + 1)
    return gridDim, gridBlock

""" Computation functions.

Offload computation efficiently on respective device.
"""
def matmul(a, b, method='cpu'):
    if method == 'cpu':
        return cpu_matmul(a, b)
    elif method == 'gpu':
        m, n, k = a.shape[0], a.shape[1], b.shape[1]
        c = np.zeros(shape=(m, k))
        gridDim, gridBlock = get_cuda_execution_config(m, k)
        numba_matmul[gridDim, gridBlock](a, b, c, m, n, k)
        return c
    else:
        raise UserWarning('Unknown computation method.')

def matsum(a, b, method='cpu'):
    if method == 'cpu':
        return cpu_matsum(a, b)
    if len(a.shape) > 1:
        m, n = a.shape[0], a.shape[1]
    else:
        m, n = a.shape[0], 1
        a = a.reshape(m, n)
    if a.shape[0] == b.shape[0] and len(a.shape) != len(b.shape):
        b = b.reshape(a.shape[0], 1)
    if a.shape != b.shape:
        b = np.repeat(b, a.shape[0], axis=0)
        b = b.reshape(a.shape)
    if method == 'gpu':
        m, n = a.shape[0], a.shape[1]
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_matsum[gridDim, gridBlock](a, b, c, m, n)
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    else:
        raise UserWarning('Unknown computation method.')

def matprod(a, b, method='cpu'):
    if method == 'cpu':
        return cpu_matprod(a, b)
    if len(a.shape) > 1:
        m, n = a.shape[0], a.shape[1]
    else:
        m, n = a.shape[0], 1
        a = a.reshape(m, n)
    if a.shape[0] == b.shape[0] and len(a.shape) != len(b.shape):
        b = b.reshape(a.shape[0], 1)
    if a.shape != b.shape:
        b = np.repeat(b, a.shape[0], axis=0)
        b = b.reshape(a.shape)
    if method == 'gpu':
        m, n = a.shape[0], a.shape[1]
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_matprod[gridDim, gridBlock](a, b, c, m, n)
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    else:
        raise UserWarning('Unknown computation method.')

def elemwise_sum(a, value, method='cpu'):
    if method == 'cpu':
        return cpu_elemwise_sum(a, value)
    if len(a.shape) > 1:
        m, n = a.shape[0], a.shape[1]
    else:
        m, n = a.shape[0], 1
        a = a.reshape(m, n)
    if method == 'gpu':
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_elemwise_sum[gridDim, gridBlock](a, value, c, m, n)
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    else:
        raise UserWarning('Unknown computation method.')

def elemwise_prod(a, value, method='cpu'):
    if method == 'cpu':
        return cpu_elemwise_prod(a, value)
    if len(a.shape) > 1:
        m, n = a.shape[0], a.shape[1]
    else:
        m, n = a.shape[0], 1
        a = a.reshape(m, n)
    if method == 'gpu':
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_elemwise_prod[gridDim, gridBlock](a, value, c, m, n)
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    else:
        raise UserWarning('Unknown computation method.')

def elemwise_max(a, value, method='cpu'):
    if method == 'cpu':
        return cpu_elemwise_max(a, value)
    if len(a.shape) > 1:
        m, n = a.shape[0], a.shape[1]
    else:
        m, n = a.shape[0], 1
        a = a.reshape(m, n)
    if method == 'gpu':
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_elemwise_max[gridDim, gridBlock](a, value, c, m, n)
        return c
    else:
        raise UserWarning('Unknown computation method.')