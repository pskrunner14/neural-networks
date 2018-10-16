""" FUNCTIONAL API: functions.py
*   Purpose: Core functional API for performing computation on CPU/GPU.
*   @author Prabhsimran Singh
*   @version 2.0 17/10/18
"""
import numpy as np

from ops.cpu_ops import (
    cpu_matmul,
    cpu_matsum,
    cpu_matprod,
    cpu_sum,
    cpu_prod,
    cpu_maximum
)
from ops.numba_ops import (
    numba_matmul,
    numba_matsum,
    numba_matprod,
    numba_sum,
    numba_prod,
    numba_maximum
)

NUM_THREADS = 32

def get_cuda_execution_config(m, n):
    gridBlock = (NUM_THREADS, NUM_THREADS)
    gridDim = ((n // gridBlock[0]) + 1, (m // gridBlock[1]) + 1)
    return gridDim, gridBlock

def matmul(a, b, method='cpu'):
    # fall back to cpu if dim inconsistency (numpy handle)
    if method == 'cpu' or len(a.shape) != len(b.shape) or len(a.shape) == 1 or len(b.shape) == 1:
        return cpu_matmul(a, b)
    elif method == 'gpu':
        m, n, k = a.shape[0], a.shape[1], b.shape[1]
        c = np.zeros(shape=(m, k))
        gridDim, gridBlock = get_cuda_execution_config(m, k)
        numba_matmul[gridDim, gridBlock](a, b, c, m, n, k)
        return c

def matsum(a, b, method='cpu'):
    if method == 'cpu' or len(a.shape) != len(b.shape) or len(a.shape) == 1 or len(b.shape) == 1:
        return cpu_matsum(a, b)
    if method == 'gpu':
        m, n = a.shape[0], a.shape[1]
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_matsum[gridDim, gridBlock](a, b, c, m, n)
        return c.reshape((m, n))

def matprod(a, b, method='cpu'):
    if method == 'cpu' or len(a.shape) != len(b.shape) or len(a.shape) == 1 or len(b.shape) == 1:
        return cpu_matprod(a, b)
    if method == 'gpu':
        m, n = a.shape[0], a.shape[1]
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_matprod[gridDim, gridBlock](a, b, c, m, n)
        return c.reshape((m, n))

def sum(a, value, method='cpu'):
    if method == 'cpu' or len(a.shape) == 1:
        return cpu_sum(a, value)
    if method == 'gpu':
        m, n = a.shape[0], a.shape[1]
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_sum[gridDim, gridBlock](a, value, c, m, n)
        return c.reshape((m, n))

def prod(a, value, method='cpu'):
    if method == 'cpu' or len(a.shape) == 1:
        return cpu_prod(a, value)
    if method == 'gpu':
        m, n = a.shape[0], a.shape[1]
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_prod[gridDim, gridBlock](a, value, c, m, n)
        return c.reshape((m, n))

def maximum(a, value, method='cpu'):
    if method == 'cpu' or len(a.shape) == 1:
        return cpu_maximum(a, value)
    if method == 'gpu':
        m, n = a.shape[0], a.shape[1]
        c = np.zeros(shape=(m, n))
        gridDim, gridBlock = get_cuda_execution_config(m, n)
        numba_maximum[gridDim, gridBlock](a, value, c, m, n)
        return c