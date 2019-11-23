from Utils import readDataWine
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    x, y = readDataWine("datasets/wine.data")
    x = x[:5]
    y = y[:5]
    print(x)
    print(y)

    nn = NeuralNetwork(0.1)
    nn.readNetworkArchitecture("testfiles/wineNetwork.txt")
    nn.train(x.values.tolist(), [[i] for i in y.tolist()], 5)
    print("Nice")
