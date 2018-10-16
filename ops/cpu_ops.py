""" NUMPY CPU API: cpu_ops.py
*   Purpose: NumPy API for performing computation on the CPU.
*   @author Prabhsimran Singh
*   @version 1.0 17/10/18
"""
import numpy as np

def cpu_matmul(a, b):
    return np.dot(a, b)

def cpu_matsum(a, b):
    return np.add(a, b)

def cpu_matprod(a, b):
    return np.multiply(a, b)

def cpu_sum(a, value):
    return cpu_matsum(a, value)

def cpu_prod(a, value):
    return cpu_matprod(a, value)

def cpu_maximum(a, value):
    return np.maximum(a, value)