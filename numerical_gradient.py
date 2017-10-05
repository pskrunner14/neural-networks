import numpy as np
from numpy import e
from numpy.linalg import norm

class Neural_Network(object):

    def __init__(self):
        #Define HyperParameters
        self.inputLayerSize  = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        #Weights (Parameters) - Arrays containing random values
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        #Propagation inputs through
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
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

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    """
    NUMERICAL GRADIENT CHECKING
    """

    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        #ravel means flatter so that all elements of matrix are in one row
        #and then concat the two gradient matrices
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    epsilon = e - 4

    for i in range(len(paramsInitial)):
        #set perturbation vector
        perturb[i] = epsilon
        
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        #compute the numerical gradient
        numgrad[i] = (loss2 - loss1) / (2 * epsilon)

        #return the value we changed back to zero
        perturb[i] = 0

    #return params to original value
    N.setParams(paramsInitial)

    return numgrad

def main():
    # X = (hours sleeping, hours studying), y = Score on test
    X = np.array(([3,5], [5,1], [10,2]), dtype = float)
    y = np.array(([75], [82], [93]), dtype = float)

    # Normalize
    X = X / np.amax(X, axis = 0)
    y = y / 100 #Max test score is 100

    network = Neural_Network()

    numgrad = computeNumericalGradient(network, X, y)
    grad = network.computeGradients(X, y)

    print(numgrad)
    print(grad)

    print(norm(grad - numgrad) / norm(grad + numgrad))

if __name__ == '__main__':
    main()
