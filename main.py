from __future__ import print_function
import argparse

import numpy as np

from nn.model import Model
from nn.layers import Dense
from nn.data import load_dataset

np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--epochs', type=int, default=10, dest='epochs',
                        help='Number of iterations for training')
    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size',
                        help='Batch size for one epoch in training')
    parser.add_argument('--lr', type=float, default=0.001, dest='lr',
                        help='Initial learning rate')
    return parser.parse_args()


def main():
    args = parse_args()

    # dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

    # model layer dimensions
    input_dim = X_train.shape[1]
    num_classes = 10

    # create model
    model = Model()
    model.add(Dense(input_dim, 100), activation='relu')
    model.add(Dense(100, 200), activation='relu')
    model.add(Dense(200, 200), activation='relu')
    model.add(Dense(200, num_classes))

    # train model
    model.fit(X_train, y_train, val_data=(X_val, y_val), verbose=True,
              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # evaluate model
    model.eval(X_test, y_test, verbose=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')
