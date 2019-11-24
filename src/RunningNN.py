from Utils import readDataWine, OneHotEncoding
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    x, y = readDataWine("datasets/wine.data")

    print(x)
    print(y)

    nn = NeuralNetwork(0.1)
    nn.readNetworkArchitecture("testfiles/wineNetwork.txt")
    # nn.numericalCheck(x.values.tolist(), y)
    nn.train(x.values.tolist(), y, 5000, saveResults = "results/testresults.txt", saveResultsEveryInterval = 1)
    print("Nice")
