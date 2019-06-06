from __future__ import print_function
import os

import numpy as np
from autograd import elementwise_grad as grad

from . import ReLU
from . import (
    softmax_crossentropy_with_logits,
    grad_softmax_crossentropy_with_logits
)
from . import iterate_minibatches

np.random.seed(42)


class Model():

    def __init__(self, name='model'):
        self.name = name
        self.__graph = []

    def add(self, layer, activation=None):
        """ Adds a layer with activation to the network.

        Args:
            layer (Layer): layer module to add to the network module list.
            activation (str): type of the activation to add after the layer.
        """
        self.__graph.append(layer)
        if activation == 'relu':
            self.__graph.append(ReLU())

    def __forward(self, X):
        """ Forward pass of the model.

        Args:
            X (numpy.ndarray): the input to the model for one forward pass.
        Returns:
            list: activations of the network layers.
        """
        activations = []
        A = X

        for layer in self.__graph:
            activations.append(layer.forward(A))
            A = activations[-1]

        assert len(activations) == len(self.__graph)
        return activations

    def compile(self, loss='softmax', optim='sgd'):
        pass

    def predict(self, X):
        """ Compute network predictions.

        Args:
            X (numpy.ndarray): the input to the model to compute y_hat.
        Returns:
            numpy.ndarray: logits for the final layer activations.
        """
        logits = self.__forward(X)[-1]
        return logits.argmax(axis=-1)

    def fit(self, X, y, val_data, **kwargs):
        epochs = kwargs.get('epochs', 20)
        verbose = kwargs.get('verbose', True)

        if verbose:
            print('Training model for {} epochs'.format(epochs))
        for epoch in range(1, epochs + 1):
            # train batch
            train_loss, train_acc = self.__fit_epoch(X, y, **kwargs)
            # val loss and accuracy
            logits_val = self.__forward(val_data[0])[-1]
            val_loss = np.mean(
                softmax_crossentropy_with_logits(logits_val, val_data[1]))
            val_acc = np.mean(np.argmax(logits_val, axis=-1) == val_data[1])
            # log info
            if verbose:
                print('Epoch[{}/{}]   train loss: {:.4f}   -   train acc: {:.4f}   -   val loss: {:.4f}   -   val acc: {:.4f}\n'
                      .format(epoch, epochs, train_loss, train_acc, val_loss, val_acc))

    def __fit_epoch(self, X, y, **kwargs):
        batch_size = kwargs.get('batch_size', 64)
        loss, acc = [], []
        for x_batch, y_batch in iterate_minibatches(X, y, batchsize=batch_size, shuffle=True):
            loss_iter, acc_iter = self.__fit_batch(x_batch, y_batch, **kwargs)
            loss.append(loss_iter)
            acc.append(acc_iter)
        return np.mean(loss), np.mean(acc)

    def __fit_batch(self, X, y, **kwargs):
        # Get the layer activations
        layer_activations = self.__forward(X)
        # layer_input[i] is an input for network[i]
        layer_inputs = [X] + layer_activations
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = softmax_crossentropy_with_logits(logits, y)
        grad_loss = grad_softmax_crossentropy_with_logits(logits, y)

        acc = (self.predict(X) == y)

        # Backpropagate the gradients to all layers
        for l in range(len(self.__graph))[::-1]:
            grad_loss = self.__graph[l].backward(layer_inputs[l], grad_loss, **kwargs)

        return np.mean(loss), np.mean(acc)

    def eval(self, X, y, **kwargs):
        verbose = kwargs.get('verbose', True)
        if verbose:
            print('Evaluating model on {} samples'.format(X.shape[0]))
        # eval loss and accuracy
        logits = self.__forward(X)[-1]
        loss = np.mean(softmax_crossentropy_with_logits(logits, y))
        acc = np.mean(np.argmax(logits, axis=-1) == y)
        # log info
        print('eval loss: {:.4f}   -   eval acc: {:.4f}\n'
                  .format(loss, acc))
