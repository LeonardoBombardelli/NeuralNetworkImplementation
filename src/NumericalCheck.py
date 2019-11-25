import argparse
import numpy as np

from NeuralNetwork import NeuralNetwork


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-architecture', nargs=1, type=str, required=True)
    parser.add_argument('-weights', nargs=1, type=str, required=False)
    parser.add_argument('-x', nargs=1, type=str, required=True)
    parser.add_argument('-y', nargs=1, type=str, required=True)
    arguments = parser.parse_args()

    nn = NeuralNetwork(0.1)
    nn.readNetworkArchitecture(arguments.architecture[0])

    if arguments.weights is not None:
        nn.readNetworkWeights(arguments.weights[0])

    xs = np.loadtxt(arguments.x[0])
    ys = np.loadtxt(arguments.y[0])

    if len(xs.shape) == 0:
        xs = np.asarray([xs])
        ys = np.asarray([ys])

    xs = np.asarray([[x] for x in xs])
    ys = np.asarray([[y] for y in ys])

    nn.numericalCheck(xs, ys)
