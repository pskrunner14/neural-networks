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
from ops.cuda_c_ops import (
    cuda_matmul,
    cuda_matsum,
    cuda_matprod,
    cuda_elemwise_sum,
    cuda_elemwise_prod,
    cuda_elemwise_max
)

""" Computation functions.

Offload computation efficiently on respective device.
"""
def matmul(a, b, method='cpu'):
    if method == 'cpu':
        return cpu_matmul(a, b)
    elif method == 'cuda_c':
        a = a.astype('float32')
        b = b.astype('float32')
        m, n, k = a.shape[0], a.shape[1], b.shape[1]
        a = a.flatten()
        b = b.flatten()
        c = np.zeros(shape=(m * k), dtype=np.float32)
        cuda_matmul(a, b, c, m, n, k)
        assert not np.isnan(np.sum(c)), 'mat mul is buggy'
        c = c.astype('float64')
        return c.reshape((m, k))
    elif method == 'numba':
        c = np.zeros(shape=(a.shape[0], b.shape[1]), dtype=np.float32)
        numba_matmul(a, b, c)
        return c
    else:
        raise UserWarning('Unknown computation method.')

def matsum(a, b, method='cpu'):
    if method == 'cpu':
        return cpu_matsum(a, b)
    elif method == 'cuda_c':
        a = a.astype('float32')
        b = b.astype('float32')
        if len(a.shape) > 1:
            m, n = a.shape[0], a.shape[1]
        else:
            m, n = a.shape[0], 1
        a = a.flatten()
        if len(a.shape) != len(b.shape):
            b = np.repeat(b, m, axis=0)
        b = b.flatten()
        c = np.zeros_like(a, dtype=np.float32)
        cuda_matsum(a, b, c, m, n)
        assert not np.isnan(np.sum(c)), 'mat sum is buggy'
        c = c.astype('float64')
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    elif method == 'numba':
        return numba_matsum(a, b)
    else:
        raise UserWarning('Unknown computation method.')

def matprod(a, b, method='cpu'):
    if method == 'cpu':
        return cpu_matprod(a, b)
    elif method == 'cuda_c':
        a = a.astype('float32')
        b = b.astype('float32')
        if len(a.shape) > 1:
            m, n = a.shape[0], a.shape[1]
        else:
            m, n = a.shape[0], 1
        a = a.flatten()
        b = b.flatten()
        c = np.zeros_like(a, dtype=np.float32)
        cuda_matprod(a, b, c, m, n)
        assert not np.isnan(np.sum(c)), 'mat prod is buggy'
        c = c.astype('float64')
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    elif method == 'numba':
        return numba_matprod(a, b)
    else:
        raise UserWarning('Unknown computation method.')

def elemwise_sum(a, value, method='cpu'):
    if method == 'cpu':
        return cpu_elemwise_sum(a, value)
    elif method == 'cuda_c':
        a = a.astype('float32')
        if len(a.shape) > 1:
            m, n = a.shape[0], a.shape[1]
        else:
            m, n = a.shape[0], 1
        a = a.flatten()
        c = np.zeros_like(a, dtype=np.float32)
        cuda_elemwise_sum(a, value, c, m, n)
        assert not np.isnan(np.sum(c)), 'element-wise sum is buggy'
        c = c.astype('float64')
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    elif method == 'numba':
        return numba_elemwise_sum(a, value)
    else:
        raise UserWarning('Unknown computation method.')

def elemwise_prod(a, value, method='cpu'):
    if method == 'cpu':
        return cpu_elemwise_prod(a, value)
    elif method == 'cuda_c':
        a = a.astype('float32')
        if len(a.shape) > 1:
            m, n = a.shape[0], a.shape[1]
        else:
            m, n = a.shape[0], 1
        a = a.flatten()
        c = np.zeros_like(a, dtype=np.float32)
        cuda_elemwise_prod(a, value, c, m, n)
        assert not np.isnan(np.sum(c)), 'element-wise prod is buggy'
        c = c.astype('float64')
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    elif method == 'numba':
        return numba_elemwise_prod(a, value)
    else:
        raise UserWarning('Unknown computation method.')

def elemwise_max(a, value, method='cpu'):
    if method == 'cpu':
        return cpu_elemwise_max(a, value)
    elif method == 'cuda_c':
        a = a.astype('float32')
        if len(a.shape) > 1:
            m, n = a.shape[0], a.shape[1]
        else:
            m, n = a.shape[0], 1
        a = a.flatten()
        c = np.zeros_like(a, dtype=np.float32)
        cuda_elemwise_max(a, value, c, m, n)
        assert not np.isnan(np.sum(c)), 'element-wise max is buggy'
        c = c.astype('float64')
        return c.reshape((m, )) if n == 1 else c.reshape((m, n))
    elif method == 'numba':
        return numba_elemwise_max(a, value)
    else:
        raise UserWarning('Unknown computation method.')