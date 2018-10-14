from __future__ import print_function

import autograd.numpy as np

from autograd import elementwise_grad as grad

np.random.seed(42)

def softmax_crossentropy_with_logits(logits, targets):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    logits_for_answers = logits[np.arange(len(logits)), targets]
    xentropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy

def grad_softmax_crossentropy_with_logits(logits, targets):
    grad_softmax = grad(softmax_crossentropy_with_logits)
    return grad_softmax(logits, targets)