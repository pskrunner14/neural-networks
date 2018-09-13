from __future__ import print_function
import numpy as np
np.random.seed(42)

class Losses():

    @staticmethod
    def softmax_crossentropy_with_logits(logits,reference_answers):
        """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
        logits_for_answers = logits[np.arange(len(logits)),reference_answers]
        
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
        
        return xentropy

    @staticmethod
    def grad_softmax_crossentropy_with_logits(logits,reference_answers):
        """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)),reference_answers] = 1
        
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        
        return (- ones_for_answers + softmax) / logits.shape[0]