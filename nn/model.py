from __future__ import print_function
import os

import numpy as np
from autograd import elementwise_grad as grad

from . import (
    softmax_crossentropy_with_logits,
    grad_softmax_crossentropy_with_logits
)
from . import iterate_minibatches

np.random.seed(42)

class Model():

    def __init__(self, name='model'):
        self.name = name
        self.__network = []

    def add(self, layer): 
        self.__network.append(layer)

    def __forward(self, X):
        """ Compute activations of all network layers by 
        applying them sequentially. Return a list of activations 
        for each layer. Make sure last activation corresponds to network logits.
        """
        activations = []
        A = X
        
        for layer in self.__network:
            activations.append(layer.forward(A))
            A = activations[-1]
            
        assert len(activations) == len(self.__network)
        return activations
        
    def predict(self, X):
        """ Compute network predictions.

        """
        logits = self.__forward(X)[-1]
        return logits.argmax(axis=-1)

    def fit(self, X, y, val_data, verbose=True, **kwargs):
        epochs = kwargs.get('epochs', 20)
        for epoch in range(1, epochs + 1):
            # train batch
            train_loss, train_acc = self.__fit_epoch(X, y, **kwargs)
            
            # val loss and accuracy
            logits_val = self.__forward(val_data[0])[-1]
            val_loss = np.mean(softmax_crossentropy_with_logits(logits_val, val_data[1]))
            val_acc = np.mean(self.predict(val_data[0])==val_data[1])

            if verbose:
                print('Epoch[{}/{}]   train loss: {:.4f}   -   train acc: {:.4f}   -   val loss: {:.4f}   -   val acc: {:.4f}\n'
                    .format(epoch, epochs, train_loss, train_acc, val_loss, val_acc))

    def eval(self, X, y, verbose=True):
        print('Evaluating model on {} samples'.format(X.shape[0]))
        logits = self.__forward(X)[-1]
        loss = np.mean(softmax_crossentropy_with_logits(logits, y))
        acc = np.mean(self.predict(X)==y)
        if verbose:
            print('eval loss: {:.4f}   -   eval acc: {:.4f}\n'.format(loss, acc))

    def __fit_epoch(self, X, y, **kwargs):
        batch_size = kwargs.get('batch_size', 64)
        loss = []
        acc = []
        for x_batch, y_batch in iterate_minibatches(X, y, batchsize=batch_size, shuffle=True):
            loss_iter, acc_iter = self.__fit_batch(x_batch, y_batch, **kwargs)
            loss.append(loss_iter)
            acc.append(acc_iter)
        return np.mean(loss), np.mean(acc)

    def __fit_batch(self, X, y, **kwargs):
        # Get the layer activations
        layer_activations = self.__forward(X)
        layer_inputs = [X] + layer_activations  #layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = softmax_crossentropy_with_logits(logits, y)
        grad_loss = grad_softmax_crossentropy_with_logits(logits, y)

        # Backpropagate the gradients to all layers
        for l in range(len(self.__network))[::-1]:
            grad_loss = self.__network[l].backward(layer_inputs[l], grad_loss, **kwargs)

        acc = (self.predict(X)==y)
        return np.mean(loss), np.mean(acc)