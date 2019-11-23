import numpy as np 
import pandas as pd 

class NeuralNetworkNP():
    def __init__(self, architecture, verbose = False, regularizationFactor = 0.0, initialWeights = None,
    alpha = 0.001, batchSize = None):

        self.verbose = verbose

        # Learning Rate
        self.alpha = alpha

        # Architecture can be a string or the own list of values. If string, we open it as a file.
        # If the second is true, we must also pass to the __init__ the regularization factor
        if(isinstance(architecture, str)):
            self.readNetworkArchitecture(architecture)
        else:
            self.architecture = architecture
            self.regularizationFactor = regularizationFactor
        

        # Regularization factor, can be read from architecture
        self.regularizationFactor = regularizationFactor

    def readNetworkArchitecture(self, fileName: str):
        with open(fileName, "r") as openFile:
            readFile = openFile.read().splitlines()
            
        self.regularizationFactor = float(readFile[0])
        
        self.architecture = [int(x) for x in readFile[1:]]
