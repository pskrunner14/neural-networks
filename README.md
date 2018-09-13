# Neural Network from Scratch
This is a Neural Network built from scratch using only Numpy for numerical computation.

### Getting Started

In order to train the model you will need to install [Numpy](http://www.numpy.org/):
```
pip install numpy
```

Once you're done with that, you can start training the model:
```
python main.py --epochs 20 --lr 0.001 --batch-size 64 --plot 1
```

### Data
As this is just an example implementation we'll be using the standard benchmark MNIST dataset of handwritten digits for classification to train our model. You can either load the data using [Keras](https://keras.io/datasets/) or download it from [here](http://yann.lecun.com/exdb/mnist/). However I'll be using the Keras Dataset helper.

### Built with
* Python
* Numpy