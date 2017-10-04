import numpy as np

class Neural_Network(object):
    """
    on every layer, input is multiplied(dot product) by the weights(random) 
    on that layer and then an activation function(sigmoid) is applied to get yHat(result)
    """

    def __init__(self):
        #Define HyperParameters
        #Structure is fixed
        self.inputLayerSize  = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        #Weights (Parameters)
        #Arrays containing random values
        #W1 - weights between layers 1 and 2
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        #W2 - weights between layers 2 and 3
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        #Propagation inputs through
        # input_matrix(X) * weights(W1) {random weights} = output_layer_2(z2)
        self.z2 = np.dot(X, self.W1)
        #f(output_layer_2(Z2) {before activation}) = sigmoid(Z2) = output_layer_2(a2) {after activation}
        self.a2 = self.sigmoid(self.z2)
        #output_layer_2(a2) * weights(W2) {random weights} = output_layer_3(z3)
        self.z3 = np.dot(self.a2, self.W2)
        #f(output_layer_3(z3) {before activation}) = output_layer_3(yHat) {after activation}
        yHat    = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, Z):
        #Apply sigmoid activation function to scalar, vector or matrix
        return 1/(1 + np.exp(-Z))

def main():

    network = Neural_Network()
    input_array = np.array([
        [0.546, 0.654],
        [0.786, 0.876],
        [0.984, 0.884]
    ])
    result = network.forward(input_array)
    print("Result: " + str(result))

if __name__ == '__main__':
    main()
