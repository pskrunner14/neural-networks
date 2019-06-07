import numpy as np
np.random.seed(42)

from .data import iterate_minibatches

from .layers import (
    Layer,
    Dense
)
from .activations import (
    ReLU,
    Sigmoid,
    Softmax
)
from .optim import (
    Optim,
    SGD,
    RMSprop
)
from .loss import (
    Loss,
    CategoricalCrossentropy
)

class Model():

    def __init__(self, name='model'):
        self.name = name
        self._graph = []

    def __repr__(self):
        return repr(self._graph)

    def __str__(self):
        return str(self._graph)

    def __call__(self, X):
        return self._forward(X)[-1]

    def add(self, layer, activation=None):
        """ Adds a layer with activation to the network.

        Args:
            layer (Layer): layer module to add to the network module list.
            activation (str): type of the activation to add after the layer.
        """
        self._graph.append(layer)
        if isinstance(activation, str):
            if activation == 'relu':
                self._graph.append(ReLU())
            elif activation == 'sigmoid':
                self._graph.append(Sigmoid())
            elif activation == 'softmax':
                self._graph.append(Softmax())
        else:
            assert isinstance(activation, Layer), \
                   'activation is not an nn.layers.Layer object'

    def _forward(self, X):
        """ Forward pass of the model.

        Args:
            X (numpy.ndarray): the input to the model for one forward pass.
        Returns:
            list: activations of the network layers.
        """
        activations = []
        A = X

        for layer in self._graph:
            activations.append(layer.forward(A))
            A = activations[-1]

        assert len(activations) == len(self._graph)
        return activations

    def compile(self, optim='sgd', loss='softmax'):
        if isinstance(optim, str):
            if optim == 'sgd':
                self._optim = SGD()
            elif optim == 'rmsprop':
                self._optim = RMSprop()
        else:
            assert isinstance(optim, Optim), \
                   'optimizer is not an nn.optim.Optim object'

        trainable_vars = []
        for layer in self._graph:
            if layer.type == 'layer':
                trainable_vars.extend(layer.trainable_vars)
        for var in trainable_vars:
            assert var.requires_grad == True, 'trainable variable does not support differentiation'
        self._optim.vars = trainable_vars

        if isinstance(loss, str):
            if loss == 'categorical_crossentropy':
                self._loss = CategoricalCrossentropy()
        else:
            assert isinstance(loss, Loss), \
                   'loss is not an nn.loss.Loss object'

    def predict(self, X):
        """ Compute network predictions.

        Args:
            X (numpy.ndarray): the input to the model to compute y_hat.
        Returns:
            numpy.ndarray: logits for the final layer activations.
        """
        probs = self._forward(X)[-1]
        return np.argmax(probs, axis=1)

    def fit(self, X, y, val_data, **kwargs):
        epochs = kwargs.get('epochs', 20)
        verbose = kwargs.get('verbose', True)

        if verbose:
            print('Training model for {} epochs'.format(epochs))
        for epoch in range(1, epochs + 1):
            # train batch
            train_loss, train_acc = self._fit_epoch(X, y, **kwargs)
            # val loss and accuracy
            logits_val = self._forward(val_data[0])[-1]
            val_loss = np.mean(self._loss(logits_val, val_data[1]))
            val_acc = np.mean(np.argmax(logits_val, axis=-1) == val_data[1])
            # log info
            if verbose:
                print('Epoch[{}/{}]   train loss: {:.4f}   -   train acc: {:.4f}   -   val loss: {:.4f}   -   val acc: {:.4f}\n'
                      .format(epoch, epochs, train_loss, train_acc, val_loss, val_acc))

    def _fit_epoch(self, X, y, **kwargs):
        batch_size = kwargs.get('batch_size', 64)
        loss, acc = [], []
        for x_batch, y_batch in iterate_minibatches(X, y, batchsize=batch_size, shuffle=True):
            loss_iter, acc_iter = self._fit_batch(x_batch, y_batch, **kwargs)
            loss.append(loss_iter)
            acc.append(acc_iter)
        return np.mean(loss), np.mean(acc)

    def _fit_batch(self, X, y, **kwargs):
        # Get the layer activations
        layer_activations = self._forward(X)
        probs = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = self._loss(probs, y)
        grad_loss = self._loss.backward(y)

        # Accumulate the gradients in all layer vars
        for i in range(len(self._graph))[::-1]:
            grad_loss = self._graph[i].backward(grad_loss)
        # Update the vars using the gradients
        self._optim.step()

        acc = np.argmax(probs, axis=1) == y
        return np.mean(loss), np.mean(acc)

    def eval(self, X, y, **kwargs):
        verbose = kwargs.get('verbose', True)
        if verbose:
            print('Evaluating model on {} samples'.format(X.shape[0]))
        # eval loss and accuracy
        probs = self._forward(X)[-1]
        loss = np.mean(self._loss(probs, y))
        acc = np.mean(np.argmax(probs, axis=1) == y)
        # log info
        print('eval loss: {:.4f}   -   eval acc: {:.4f}\n'
                  .format(loss, acc))
