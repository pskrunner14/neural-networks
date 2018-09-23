import numpy as np
import ctypes

from ctypes import POINTER, c_float, c_int

# Build shared object file using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_mat_ops.so ops/mat_ops.cu
# extract cuda function pointers in the shared object cuda_mat_ops.so
dll = ctypes.CDLL('./lib/cuda_mat_ops.so', mode=ctypes.RTLD_GLOBAL)

def get_cuda_mat_sum(dll):
    func = dll.cuda_mat_sum
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int]
    return func

def get_cuda_mat_prod(dll):
    func = dll.cuda_mat_prod
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int]
    return func

def get_cuda_mat_mul(dll):
    func = dll.cuda_mat_mul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
    return func

__cuda_mat_sum = get_cuda_mat_sum(dll)
__cuda_mat_prod = get_cuda_mat_prod(dll)
__cuda_mat_mul = get_cuda_mat_mul(dll)

# convenient python wrappers for cuda functions

def cuda_mat_sum(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_mat_sum(a_p, b_p, c_p, m, n)

def cuda_mat_prod(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_mat_prod(a_p, b_p, c_p, m, n)

def cuda_mat_mul(a, b, c, m, n, k):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_mat_mul(a_p, b_p, c_p, m, n, k)

def get_test_params():
    size = int(16)
    a = np.array([3.0] * (size * size), dtype=np.float32)
    b = np.array([3.0] * (size * size), dtype=np.float32)
    c = np.zeros(shape=(size * size), dtype=np.float32)
    return a, b, c, size

def main():
    a, b, c, size = get_test_params()
    cuda_mat_mul(a, b, c, size, size, size)
    assert np.all(c==144.0), "Matrix dot-product operation is buggy"

if __name__ == '__main__':
    main()