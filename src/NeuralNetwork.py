

class NeuralNetwork():
    
    def readNetworkArchitecture(self, fileName):
        with open(fileName, "r") as openFile:
            readFile = openFile.read().splitlines()
        self.regularizationFactor = float(readFile[0])
        self.architecture = [int(x) for x in readFile[1:]]

if __name__ == "__main__":
    a = NeuralNetwork()
    a.readNetworkArchitecture("testfiles/network.txt")
