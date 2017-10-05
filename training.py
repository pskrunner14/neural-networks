import numpy as np
from scipy import optimize
from matplotlib.pyplot import plot
from matplotlib.pyplot import grid
from matplotlib.pyplot import xlabel
from matplotlib.pyplot import ylabel
from matplotlib.pyplot import show

from numerical_gradient import Neural_Network

class Trainer(object):

    """
    TRAINING THE NEURAL NET
    """

    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradient(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}

        #BFGS implementation
        #jac -> jacobian
        #callback will help us to track the cost function value as we train the network
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)

        #here _res.x are the weights after the optimization/ training
        self.N.setParams(_res.x)
        self.optimizationResults = _res

def main():

    # X = (hours sleeping, hours studying), y = Score on test
    X = np.array(([3,5], [5,1], [10,2]), dtype = float)
    y = np.array(([75], [82], [93]), dtype = float)

    # Normalize
    X = X / np.amax(X, axis = 0)
    y = y / 100 #Max test score is 100

    network = Neural_Network()

    trainer = Trainer(network)
    trainer.train(X, y)
    
    #plot the training data
    plot(trainer.J)
    grid(1)
    xlabel('Iterations')
    ylabel('Cost')
    show()

if __name__ == '__main__':
    main()