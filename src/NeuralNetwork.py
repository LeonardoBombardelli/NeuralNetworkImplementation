import numpy as np


class NeuralNetwork():
    def __init__(self):
        self.regularizationFactor = -1.0
        # Amount of neurons per layer
        self.architecture = []
        # Weights of the nn; the ith element contains the weights
        # from the ith layer to the following layer (i+1)
        self.weights = []
        # Activation for each neuron of each layer
        self.activation = []

    def readNetworkArchitecture(self, fileName):
        with open(fileName, "r") as openFile:
            readFile = openFile.read().splitlines()
        self.regularizationFactor = float(readFile[0])
        self.architecture = [int(x) for x in readFile[1:]]

        # First layer is set to input, when computing activation
        # Element 0 is always 1 for each layer (bias)
        for i, neurons in enumerate(self.architecture):
            self.activation.append(np.zeros(neurons+1))
            self.activation[i][0] = 1.0

        for i in range(len(self.architecture)-1):
            # Destination, origin; self.architecture[i]+1 to account for bias
            self.weights.append(np.random.rand(self.architecture[i+1],
                                               self.architecture[i]+1))

    def readNetworkWeights(self, fileName):
        with open(fileName, "r") as openFile:
            readFile = openFile.read().splitlines()

        assert len(readFile) == len(self.weights), \
               'Error: number of weights does not match network layout'

        for i, line in enumerate(readFile):
            nodeWeights = line.split(';')

            assert len(nodeWeights) == self.architecture[i+1], \
                   'Error: number of weights does not match network layout'

            for j, nodeWeight in enumerate(nodeWeights):
                actualWeights = [float(value) for value in nodeWeight.split(',')]

                assert len(actualWeights) == self.architecture[i]+1, \
                       'Error: number of weights does not match network layout'

                # Layer i, node j (all sources)
                self.weights[i][j] = actualWeights

    def computeActivations(self, input):
        assert len(input) == len(self.activation[0])-1, \
            'Error: input size is not compatible with network shape'

        self.activation[0][1:] = input
        for i in range(1, len(self.activation)):
            for j in range(1, len(self.activation[i])):
                # i-1 as we want the previous layer
                # j-1 as we want the weight from the bias (self.weights)
                self.activation[i][j] = (self.activation[i-1] * self.weights[i-1][j-1]).sum()

    def getPredictions(self):
        return self.activation[len(self.activation)-1][1:]

if __name__ == "__main__":
    a = NeuralNetwork()
    a.readNetworkArchitecture("testfiles/network.txt")
    a.readNetworkWeights("testfiles/weights.txt")
    a.computeActivations([0.6969, 0.666])
    print(a.getPredictions())
