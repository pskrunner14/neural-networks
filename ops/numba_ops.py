import numpy as np
from numba import vectorize, int64, float32, void, jit, cuda, guvectorize

"""
Numba CUDA config paths:
/usr/local/cuda-9.2/nvvm/libdevice
/usr/local/cuda-9.2/nvvm/bin
/usr/local/cuda-9.2/lib64/
"""

# @jit(void(float32[:,:],float32[:,:],float32[:,:]))
def numba_matmul(a, b, c):
    pass

# @jit(void(float32[:,:],float32[:,:],float32[:,:]))
def numba_matsum(a, b):
    pass

# @jit(void(float32[:,:],float32[:,:],float32[:,:]))
def numba_matprod(a, b):
    pass

# @jit(void(float32[:,:],float32[:,:],float32[:,:]))
def numba_elemwise_sum(a, value):
    pass

# @jit(void(float32[:,:],float32[:,:],float32[:,:]))
def numba_elemwise_prod(a, value):
    pass

# @jit(void(float32[:,:],float32[:,:],float32[:,:]))
def numba_elemwise_max(a, value):
    pass