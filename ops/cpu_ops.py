import numpy as np

def cpu_matmul(a, b):
    return np.dot(a, b)

def cpu_matsum(a, b):
    return np.add(a, b)

def cpu_matprod(a, b):
    return np.multiply(a, b)

def cpu_elemwise_sum(a, value):
    return cpu_matsum(a, value)

def cpu_elemwise_prod(a, value):
    return cpu_matprod(a, value)

def cpu_elemwise_max(a, value):
    return np.maximum(a, value)