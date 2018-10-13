import ctypes
import numpy as np

from ctypes import POINTER, c_float, c_int

# Build shared object file using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_ops.so ops/mat_ops.cu
# extract cuda function pointers in the shared object cuda_mat_ops.so
dll = ctypes.CDLL('./lib/cuda_mat_ops.so', mode=ctypes.RTLD_GLOBAL)

def get_cuda_matsum(dll):
    func = dll.cuda_matsum
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int]
    return func

def get_cuda_matprod(dll):
    func = dll.cuda_matprod
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int]
    return func

def get_cuda_matmul(dll):
    func = dll.cuda_matmul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
    return func

__cuda_matsum = get_cuda_matsum(dll)
__cuda_matprod = get_cuda_matprod(dll)
__cuda_matmul = get_cuda_matmul(dll)

# convenient python wrappers for cuda functions

def cuda_matsum(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_matsum(a_p, b_p, c_p, m, n)

def cuda_matprod(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_matprod(a_p, b_p, c_p, m, n)

def cuda_matmul(a, b, c, m, n, k):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_matmul(a_p, b_p, c_p, m, n, k)

def get_test_params():
    size = int(16)
    a = np.array([3.0] * (size * size), dtype=np.float32)
    b = np.array([3.0] * (size * size), dtype=np.float32)
    c = np.zeros(shape=(size * size), dtype=np.float32)
    return a, b, c, size

def main():
    a, b, c, size = get_test_params()
    cuda_matmul(a, b, c, size, size, size)
    assert np.all(c==144.0), "Matrix dot-product operation is buggy"

if __name__ == '__main__':
    main()