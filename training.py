import numpy as np
from numerical_gradient import Neural_Network
import scipy

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
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = scipy.optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

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
    

if __name__ == '__main__':
    main()