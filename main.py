from __future__ import print_function
import numpy as np
import argparse

from nn import Trainer
from nn import load_dataset, iterate_minibatches

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
    dims = [input_dim, 100, 200, 200, num_classes]

    # model trainer
    trainer = Trainer(dims=dims)

    # loop over epochs
    for epoch in range(1, args.epochs + 1):
        # train batch
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=args.batch_size, shuffle=True):
            trainer.fit(x_batch, y_batch, lr=args.lr)
        
        # compute train and val accuracy
        train_acc = np.mean(trainer.predict(X_train) == y_train)
        val_acc = np.mean(trainer.predict(X_val) == y_val)
        
        # log epoch acc and loss
        print("Epoch[{}/{}]  train acc: {:.4f}   -   val acc: {:.4f}".format(epoch, args.epochs, train_acc, val_acc))
    
    # test
    print('\nTesting on {} samples'.format(len(X_test)))
    accuracy = np.mean(trainer.predict(X_test) == y_test) * 100
    print('test acc: {:.4f}'.format(accuracy))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')