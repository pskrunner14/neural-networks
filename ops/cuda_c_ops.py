import ctypes
import numpy as np

from ctypes import POINTER, c_float, c_int

# Build shared object file using: nvcc -Xcompiler -fPIC -shared -o lib/cuda_c.so lib/cuda_c.cu
# extract cuda function pointers in the shared object cuda_c.so
dll = ctypes.CDLL('./lib/cuda_c.so', mode=ctypes.RTLD_GLOBAL)

def get_cuda_matmul(dll):
    func = dll.cuda_matmul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]
    return func

def get_cuda_matsum(dll):
    func = dll.cuda_matsum
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int]
    return func

def get_cuda_elemwise_prod(dll):
    func = dll.cuda_elemwise_prod
    func.argtypes = [POINTER(c_float), c_float, POINTER(c_float), c_int, c_int]
    return func

def get_cuda_elemwise_max(dll):
    func = dll.cuda_elemwise_max
    func.argtypes = [POINTER(c_float), c_float, POINTER(c_float), c_int, c_int]
    return func

__cuda_matmul = get_cuda_matmul(dll)
__cuda_matsum = get_cuda_matsum(dll)
__cuda_elemwise_prod = get_cuda_elemwise_prod(dll)
__cuda_elemwise_max = get_cuda_elemwise_max(dll)

# convenient python wrappers for cuda functions
def cuda_matmul(a, b, c, m, n, k):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_matmul(a_p, b_p, c_p, m, n, k)

def cuda_matsum(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_matsum(a_p, b_p, c_p, m, n)

def cuda_elemwise_prod(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_f = ctypes.c_float(b)
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_elemwise_prod(a_p, b_f, c_p, m, n)

def cuda_elemwise_max(a, b, c, m, n):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_f = ctypes.c_float(b)
    c_p = c.ctypes.data_as(POINTER(c_float))
    __cuda_elemwise_max(a_p, b_f, c_p, m, n)

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
    cuda_matsum(a, b, c, size, size)
    assert np.all(c==6.0), "Matrix sum operation is buggy"
    cuda_elemwise_prod(a, 2.5, c, size, size)
    assert np.all(c==7.5), "Element-wise product operation is buggy"
    cuda_elemwise_max(a, 4.0, c, size, size)
    assert np.all(c==4.0), "Element-wise max operation is buggy"

if __name__ == '__main__':
    main()