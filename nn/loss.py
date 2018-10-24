from __future__ import print_function

import autograd.numpy as np
from autograd import elementwise_grad as grad

np.random.seed(42)

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def softmax_crossentropy_with_logits(logits, targets):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    m = targets.shape[0]
    p = stable_softmax(logits)
    log_likelihood = -np.log(p[range(m), targets])
    loss = np.sum(log_likelihood) / m
    return loss

def grad_softmax_crossentropy_with_logits(logits, targets):
    grad_softmax = grad(softmax_crossentropy_with_logits)
    return grad_softmax(logits, targets)