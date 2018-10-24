# Neural Network from Scratch

[![Build Status](https://travis-ci.org/pskrunner14/neural-networks.svg?branch=master)](https://travis-ci.org/pskrunner14/neural-networks) [![Maintainability](https://api.codeclimate.com/v1/badges/c4736d207700f8b85167/maintainability)](https://codeclimate.com/github/pskrunner14/neural-networks/maintainability) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/405bf4d442c741a8822615f39c655f7f)](https://www.codacy.com/app/pskrunner14/neural-networks?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pskrunner14/neural-networks&amp;utm_campaign=Badge_Grade)

This is a Neural Network built from scratch using only Numpy for numerical computation.

## Getting Started

In order to train the model you will need to install some dependencies:

```bash
pip install -r requirements.txt
```

Once you're done with that, you can start training the model:

```bash
python main.py --epochs 20 --lr 0.001 --batch-size 64
```

## Data

As this is just an example implementation we'll be using the standard benchmark MNIST dataset of handwritten digits for classification to train our model. You can either load the data using [Keras](https://keras.io/datasets/) or download it from [here](http://yann.lecun.com/exdb/mnist/). However I'll be using the Keras Dataset helper.