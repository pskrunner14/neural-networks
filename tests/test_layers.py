import sys
sys.path.append('../')

import unittest
import numpy as np

from layers import Dense, ReLU
from util import eval_numerical_gradient

class TestLayers(unittest.TestCase):

    def test_relu_layer_NUMERICAL_GRADIENT_CHECK(self):
        x = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
        layer = ReLU()
        grads = layer.backward(x, np.ones([10, 32]) / (32 * 10))
        numeric_grads = eval_numerical_gradient(lambda x: layer.forward(x).mean(), x=x)
        
        self.assertTrue(np.allclose(grads, numeric_grads, rtol=1e-3, atol=0), 
            msg="gradient returned by your layer does not match the numerically computed gradient")

    def test_dense_layer_PARAMS(self):
        layer = Dense(128, 150)
        self.assertTrue(-0.05 < layer.weights.mean() < 0.05, 
            msg="The initial weights must have zero mean and small variance.")
        self.assertTrue(1e-3 < layer.weights.std() < 1e-1, 
            msg="If you know what you're doing, remove this assertion.")
        self.assertTrue(-0.05 < layer.biases.mean() < 0.05, 
            msg="Biases must be zero mean. Ignore if you have a reason to do otherwise.")
        
    def test_dense_layer_FORWARD(self):
        layer = Dense(3, 4)
        x = np.linspace(-1, 1, 2 * 3).reshape([2, 3])
        layer.weights = np.linspace(-1, 1, 3 * 4).reshape([3, 4])
        layer.biases = np.linspace(-1, 1, 4)
        
        self.assertTrue(np.allclose(layer.forward(x), 
            np.array([[ 0.07272727,  0.41212121,  0.75151515,  1.09090909],
                    [-0.90909091,  0.08484848,  1.07878788,  2.07272727]])))

    def test_dense_layer_NUMERICAL_GRADIENT_CHECK(self):
        x = np.linspace(-1, 1 , 10 * 32).reshape([10, 32])
        l = Dense(32, 64)
        numeric_grads = eval_numerical_gradient(lambda x: l.forward(x).sum(), x)
        grads = l.backward(x, np.ones([10, 64]), lr=0, alpha=0.9, epsilon=1e-8)
        assert np.allclose(grads, numeric_grads, rtol=1e-3, atol=0), "input gradient does not match numeric grad"
        print("Well done!")

    def test_dense_layer_GRADIENT_WRT_PARAMS(self):
        def compute_out_given_wb(w, b):
            layer = Dense(32, 64)
            layer.weights = np.array(w)
            layer.biases = np.array(b)
            x = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
            return layer.forward(x)
        def compute_grad_by_params(w, b):
            layer = Dense(32, 64)
            layer.weights = np.array(w)
            layer.biases = np.array(b)
            x = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
            layer.backward(x, np.ones([10, 64]), lr=1, alpha=0.9, epsilon=1e-8)
            return w - layer.weights, b - layer.biases
        w = np.random.randn(32, 64) * np.sqrt(2. / (32 + 64))
        b = np.zeros(64)
        numeric_dw = eval_numerical_gradient(lambda w: compute_out_given_wb(w, b).mean(0).sum(), w)
        numeric_db = eval_numerical_gradient(lambda b: compute_out_given_wb(w, b).mean(0).sum(), b)
        grad_w, grad_b = compute_grad_by_params(w, b)

        # self.assertTrue(np.allclose(numeric_dw, grad_w, rtol=1e-2, atol=0),
            # msg="weight gradient does not match numeric weight gradient")
        # self.assertTrue(np.allclose(numeric_db, grad_b, rtol=1e-2, atol=0), 
            # msg="bias gradient does not match numeric bias gradient")