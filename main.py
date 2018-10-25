from __future__ import print_function
import numpy as np
import argparse

from nn import Model
from nn import Dense, ReLU
from nn import load_dataset

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
    model.add(Dense(input_dim, 100))
    model.add(ReLU())
    model.add(Dense(100, 200))
    model.add(ReLU())
    model.add(Dense(200, 200))
    model.add(ReLU())
    model.add(Dense(200, num_classes))

    # loop over epochs
    for epoch in range(1, args.epochs + 1):
        # train batch
        train_loss = model.fit(X_train, y_train, batch_size=args.batch_size, lr=args.lr)
        
        # compute train and val accuracy
        train_acc = np.mean(model.predict(X_train) == y_train)
        val_acc = np.mean(model.predict(X_val) == y_val)
        
        # log epoch acc and loss
        print("Epoch[{}/{}]   train loss: {:.4f}   -   train acc: {:.4f}   -   val acc: {:.4f}".format(epoch, args.epochs, train_loss, train_acc, val_acc))
    
    # test
    print('\nTesting on {} samples'.format(len(X_test)))
    accuracy = np.mean(model.predict(X_test) == y_test) * 100
    print('test acc: {:.4f}'.format(accuracy))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')