import sys
sys.path.append('../')

import unittest
import numpy as np
from numpy.linalg import norm

from loss import (
    softmax_crossentropy_with_logits,
    grad_softmax_crossentropy_with_logits
)
from util import eval_numerical_gradient

class TestLoss(unittest.TestCase):

    def test_crossentropy_loss_NUMERICAL_GRADIENT_CHECK(self):
        logits = np.linspace(-1, 1, 500).reshape([50, 10])
        answers = np.arange(50) % 10
        softmax_crossentropy_with_logits(logits, answers)
        grads = grad_softmax_crossentropy_with_logits(logits, answers)
        numeric_grads = eval_numerical_gradient(lambda l: softmax_crossentropy_with_logits(l, answers).mean(), logits)
        self.assertTrue(np.allclose(numeric_grads, grads, rtol=1, atol=1e-8), 
            msg="The reference implementation has just failed. Someone has just changed the rules of math.")