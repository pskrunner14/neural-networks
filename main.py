from __future__ import print_function
import argparse

import numpy as np
import matplotlib.pyplot as plt

from train import Trainer
from data import iterate_minibatches, load_dataset

np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--epochs', type=int, default=20, dest='epochs',
                        help='Number of iterations for training')
    parser.add_argument('--batch-size', type=int, default=128, dest='batch_size', 
                        help='Batch size for one epoch in training')
    parser.add_argument('--lr', type=float, default=0.005, dest='lr',
                        help='Initial learning rate')
    parser.add_argument('--backend', type=str, default='cpu', dest='backend',
                        help='Type of computation backend to use [CPU/GPU]')
    return parser.parse_args()

def main():
    args = parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

    input_dim = X_train.shape[1]
    num_classes = 10
    dims = [input_dim, 1024, 1024, 256, num_classes]

    trainer = Trainer(dims=dims, backend=args.backend.lower())

    train_log = []
    val_log = []

    for epoch in range(args.epochs):
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=args.batch_size, shuffle=True):
            trainer.fit(x_batch, y_batch, lr=args.lr)
        
        train_log.append(np.mean(trainer.predict(X_train) == y_train))
        val_log.append(np.mean(trainer.predict(X_val) == y_val))
        
        print("Epoch[{}/{}]  train acc: {:.4f}   -   val acc: {:.4f}".format(epoch, args.epochs, train_log[-1], val_log[-1]))

    print('\nTesting on {} samples'.format(len(X_test)))
    accuracy = np.mean(trainer.predict(X_test) == y_test)
    print('test acc: {:.4f}'.format(accuracy))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')