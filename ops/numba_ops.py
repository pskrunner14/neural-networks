""" NUMBA GPU OPS: numba_ops.py
*   Purpose: Numba API for performing computation on the GPU.
*   @author Prabhsimran Singh
*   @version 1.0 17/10/18
"""
import numpy as np
from numba import cuda

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
def numba_sum(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] + value

@cuda.jit
def numba_prod(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] * value

@cuda.jit
def numba_maximum(a, value, c, m, n):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < m and col < n:
        c[row, col] = a[row, col] if a[row, col] > value else value