from Utils import readDataWine
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    x, y = readDataWine("datasets/wine.data")
    x = x[:5]
    y = y[:5]
    print(x)
    print(y)

    ys = []
    for i in y.tolist():
        yl = [0] * 3
        yl[i-1] = 1
        print(yl)
        ys.append(yl)

    nn = NeuralNetwork(0.1)
    nn.readNetworkArchitecture("testfiles/wineNetwork.txt")
    nn.numericalCheck(x.values.tolist(), ys)
    # nn.train(x.values.tolist(), [[i] for i in y.tolist()], 5)
    print("Nice")
