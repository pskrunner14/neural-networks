from __future__ import print_function
import argparse

import numpy as np
import matplotlib.pyplot as plt

from train import Trainer
from data import iterate_minibatches, load_dataset

np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--epochs', type=int, default=10, dest='epochs',
                        help='Number of iterations for training')
    parser.add_argument('--batch-size', type=int, default=1024, dest='batch_size', 
                        help='Batch size for one epoch in training')
    parser.add_argument('--lr', type=float, default=0.005, dest='lr',
                        help='Initial learning rate')
    parser.add_argument('--plot', type=bool, default=False, dest='plot',
                        help='Flag that indicates whether plot the accuracy during training')
    return parser.parse_args()

def main():
    args = parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

    input_dim = X_train.shape[1]
    num_classes = 10
    dims = [input_dim, 10, 20, 20, num_classes]

    trainer = Trainer(dims=dims, lr=args.lr)

    train_log = []
    val_log = []

    for epoch in range(args.epochs):
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=args.batch_size, shuffle=True):
            trainer.fit(x_batch, y_batch)
        
        train_log.append(np.mean(trainer.predict(X_train) == y_train))
        val_log.append(np.mean(trainer.predict(X_val) == y_val))
        
        print("Epoch ", epoch + 1)
        print("Train accuracy: {:.2f}%".format(train_log[-1] * 100))
        print("Val accuracy: {:.2f}%".format(val_log[-1] * 100))
        if args.plot:
            plt.plot(train_log,label='train accuracy')
            plt.plot(val_log,label='val accuracy')
            plt.legend(loc='best')
            plt.grid()
            plt.show()

    print('\nTesting on {} samples'.format(len(X_test)))
    accuracy = np.mean(trainer.predict(X_test) == y_test) * 100
    print('Test accuracy: {:.2f}%\n'.format(accuracy))

    trainer.save_model()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')