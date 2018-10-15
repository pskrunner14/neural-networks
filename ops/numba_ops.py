import numpy as np
from numba import cuda

"""
Numba CUDA config paths:
/usr/local/cuda-9.2/nvvm/libdevice
/usr/local/cuda-9.2/nvvm/bin
/usr/local/cuda-9.2/lib64/
"""

@cuda.jit
def numba_matmul(a, b, c, m, n, k):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < m and col < k:
        summ = 0
        for i in range(n):
            summ += a[row, i] * b[i, col]
        c[row, col] = summ
        
@cuda.jit
def numba_matsum(a, b, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < m and col < n:
        c[row, col] = a[row, col] + b[row, col]

@cuda.jit
def numba_matprod(a, b, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < m and col < n:
        c[row, col] = a[row, col] * b[row, col]

@cuda.jit
def numba_elemwise_sum(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < m and col < n:
        c[row, col] = a[row, col] + value

@cuda.jit
def numba_elemwise_prod(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < m and col < n:
        c[row, col] = a[row, col] * value

@cuda.jit
def numba_elemwise_max(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < m and col < n:
        c[row, col] = a[row, col] if a[row, col] > value else value