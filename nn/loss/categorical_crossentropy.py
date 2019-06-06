import numpy as np
np.random.seed(42)

from .loss import Loss

class CategoricalCrossentropy(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, probs, targets):
        pass

    def backward(self):
        pass

# def softmax_crossentropy_with_logits(logits, targets):
#     """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
#     m = targets.shape[0]
#     p = stable_softmax(logits)
#     log_likelihood = -np.log(p[range(m), targets])
#     loss = np.sum(log_likelihood) / m
#     return loss

# def grad_softmax_crossentropy_with_logits(logits, targets):
#     grad_softmax = grad(softmax_crossentropy_with_logits)
#     return grad_softmax(logits, targets)