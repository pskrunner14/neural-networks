import numpy as np

class Neural_Network(object):

    #same as forward propagation
    def __init__(self):
        self.inputLayerSize  = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    #same as forward propagation
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    """
    GRADIENT DESCENT
    """

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        #easy way to calculate errors is : 
        # let e -> error, then totalE = e1^2 + e2^2 + ...
        # where e is (y - yHat)
        # no reason to multiply with 0.5
        J = 0.5 * sum((y - self.yHat) ** 2)
        # training a network means minimizing this cost function's value i.e. J
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2

        #Get yHat form the forward method for (y - yHat)
        self.yHat = self.forward(X)

        #Calculate delta3 from scalar multiplication = (-(y - yHat) * del(z3)/del(W2))
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        #Calculate the cost wrt W2 by taking dot product of the delta3 and transpose of the second layer activity a2
        dJdW2 = np.dot(self.a2.T, delta3)

        #Similarly Calculate for cost wrt W1 with a bit of change
        #As eqn is = delta3 * W2.T * f'(Z2) * (del(Z2) / del(W1))
        #And here (del(Z2) / del(W1)) = X.T because X*W1 = Z2
        #So Matrix X will be transposed
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        #As eqn is = delta3 * W2.T * f'(Z2) * X.T = delta2 * X.T
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def sigmoid(self, Z):
        #Apply sigmoid activation function to scalar, vector or matrix
        return 1 / (1 + np.exp(-Z))

    # derivative of sigmoid function
    def sigmoidPrime(self,Z):
        #Gradient of sigmoid
        return np.exp(-Z) / ((1 + np.exp(-Z)) ** 2)

def main():

    network = Neural_Network()

    input_array = np.array([
        [0.546, 0.654],
        [0.786, 0.876],
        [0.984, 0.884]
    ])

    sample_output_array = np.array([
        [0.78],
        [0.89],
        [0.71]
    ])

    dJdW1, dJdW2 = network.costFunctionPrime(input_array, sample_output_array)

    print("dJdW1: ", dJdW1)
    print("dJdW2: ", dJdW2)

    #Now we know which way we should move our weights in
    #order to minimize our costs, hence we backpropagate
    scalar =  3

    cost1 = network.costFunction(input_array, sample_output_array)

    network.W1 += scalar * dJdW1
    network.W2 += scalar * dJdW2

    cost2 = network.costFunction(input_array, sample_output_array)

    dJdW1, dJdW2 = network.costFunctionPrime(input_array, sample_output_array)

    network.W1 -= scalar * dJdW1
    network.W2 -= scalar * dJdW2

    cost3 = network.costFunction(input_array, sample_output_array)

    print("Cost 1: ", cost1)
    print("Cost 2: ", cost2)
    print("Cost 3: ", cost3)

if __name__ == '__main__':
    main()
