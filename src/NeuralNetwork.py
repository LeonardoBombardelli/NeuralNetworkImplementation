import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

class NeuralNetwork():
    def __init__(self, alpha):
        # Set to false for now
        self.verbose = False
        self.alpha = alpha
        self.regularizationFactor = -1.0
        # Amount of neurons per layer
        self.architecture = []
        # Weights of the nn; the ith element contains the weights
        # from the ith layer to the following layer (i+1)
        self.weights = []
        # Activation for each neuron of each layer
        self.activation = []
        # Delta for each neuron of each layer
        self.delta = []
        # Gradient for each weight of each layer
        self.gradient = []

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

            # No delta for input layer; also, no delta for biases
            if i != 0:
                self.delta.append(np.zeros(neurons))

        for i in range(len(self.architecture)-1):
            # Destination, origin; self.architecture[i]+1 to account for bias
            layer = np.random.rand(self.architecture[i+1],
                                   self.architecture[i]+1)
            layer = (layer - 0.5) * 2
            self.weights.append(layer)
            self.gradient.append(np.empty((self.architecture[i+1],
                                          self.architecture[i]+1)))

        # self.architecture = np.asarray(self.architecture)
        # self.activation = np.asarray(self.activation)
        # self.weights = np.asarray(self.weights)
        # self.delta = np.asarray(self.delta)
        # self.gradient = np.asarray(self.gradient)


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

    # Single element
    def computeActivations(self, input, eps=None):
        assert len(input) == len(self.activation[0])-1, \
            'Error: input size is not compatible with network shape'

        self.activation[0][1:] = input
        for i in range(1, len(self.activation)):
            if eps is None:
                self.activation[i][1:] = (self.activation[i-1] * self.weights[i-1]).sum(1)
            else:
                self.activation[i][1:] = (self.activation[i-1] * (self.weights[i-1] + eps[i-1])).sum(1)
            self.activation[i][1:] = sigmoid(self.activation[i][1:])

    # Single element
    def computeDeltas(self, x, y):
        self.computeActivations(x); out = self.getPredictions()
        self.delta[len(self.delta)-1] = out - y

        # Reverse loop, from last hidden layer to the first one
        for i in range(len(self.delta)-2, -1, -1):
            # Skips bias activation
            self.delta[i] = ((self.delta[i+1] * self.weights[i+1][:,1:]) * \
                            self.activation[i+1][1:] * \
                            (np.ones(len(self.activation[i+1][1:])) - self.activation[i+1][1:]))[0]

    def setGradientZero(self):
        for g in self.gradient:
            g.fill(0.0)

    # Single element
    def updateGradient(self, x, y):
        self.computeDeltas(x, y)

        if self.verbose:
            print('Weights:')
            print(self.weights)
            print('Activation:')
            print(self.activation)
            print('Delta:')
            print(self.delta)

        for i in range(len(self.gradient)):
            # Delta already starts in the first hidden layer
            self.gradient[i] += self.delta[i] * self.activation[i]

    # n is the number of training instances
    def gradientRegularization(self, n):
        regularization = np.asarray(self.regularizationFactor * self.weights)

        for layer in regularization:
            layer[:,0] = 0

        if self.verbose:
            print('Gradient:')
            print(self.gradient)

        self.gradient = (1.0 / n) * (self.gradient + regularization)

        if self.verbose:
            print('Regularization:')
            print(regularization)
            print('Gradient after Regularization:')
            print(self.gradient)

    def updateTheta(self):
        for i in range(len(self.weights)):
            self.weights -= self.alpha * self.gradient

    # Batch
    def computeCost(self, x, y, eps=None, reg=True):
        assert len(x) == len(y), "Error: number of entries and labels don't match"

        j = 0.0
        for i in range(len(x)):
            self.computeActivations(x[i], eps)
            out = self.getPredictions();

            j += ([-1*v for v in y[i]] * np.log(out) - (np.ones(len(y[i])) - y[i]) * np.log(1 - out)).sum()
        j /= len(x)


        s = 0.0

        if reg:
            for w in self.weights:
                s += np.square(w[1:]).sum()
            s *= (self.regularizationFactor / (2*len(x)))

        if self.verbose:
            print('J: ', j)
            print('S: ', s)

        return j + s

    def getPredictions(self):
        return self.activation[len(self.activation)-1][1:]

    def numericalCheck(self, x, y):
        eps = 0.000005
        diff = [np.zeros_like(arr) for arr in self.gradient]
        nGradient = [np.zeros_like(arr) for arr in self.gradient]
        # print(self.gradient[0].shape)
        # print(self.gradient[1].shape)

        self.setGradientZero()
        for j in range(len(x)):
            self.updateGradient(x[j], y[j])
        print('Computed gradient:')
        print(self.gradient)

        for l in range(len(self.gradient)):
            for i in range(self.architecture[l+1]):
                for j in range(self.architecture[l]+1):
                    diff[l][i][j] = eps
                    # print('DIFF:', diff[i][0][j])
                    # print(diff)
                    J = self.computeCost(x, y, diff, False)
                    diff[l][i][j] = -eps
                    # print('DIFF:')
                    # print(diff)
                    J -= self.computeCost(x, y, diff, False)
                    diff[l][i][j] = 0.0
                    # print(i, j)
                    nGradient[l][i][j] = J / (2 * eps)

        print('DIFF:')
        print(diff)
        print('Numerical gradient:')
        print(nGradient)

    def predict(self, x):
        self.computeActivations(x)
        return self.getPredictions()

    def train(self, x, y, epochs):
        for i in range(epochs):
            print('EPOCH %d' % i)
            print('Current error: ', self.computeCost(x, y))

            self.setGradientZero()
            for j in range(len(x)):
                self.updateGradient(x[j], y[j])
            self.gradientRegularization(len(x))

            if self.verbose:
                print('Weights:')
                print(self.weights)
                print('Activation:')
                print(self.activation)
                print('Delta:')
                print(self.delta)
                print('Gradient:')
                print(self.gradient)

            self.updateTheta()

if __name__ == "__main__":
#    x = [[0.6969, 0.666]]; y = [[1.0]]
    x = [[1.5]]; y = [[1.0]]

    a = NeuralNetwork(0.1)
    a.readNetworkArchitecture("testfiles/network_numerical_test.txt")
    a.readNetworkWeights("testfiles/weights_numerical_test.txt")
    a.numericalCheck(x, y)

#    print('Before training....', a.predict([0.6969, 0.666]), '\n')
#    a.train(x, y, 1)
#    print('\n\nAfter training....', a.predict([0.6969, 0.666]))
