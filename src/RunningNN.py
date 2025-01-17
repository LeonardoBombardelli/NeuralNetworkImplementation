from Utils import readDataWine, OneHotEncoding, readDataDiabetes, readDataIonosphere, GenerateTrainAndEval, generate_kfolds
from NeuralNetwork import NeuralNetwork

def runTrain(functionToRead, datasetPath, netWorkPath, numberIterations, alpha, nClass ,saveNumber = 5):
    df = functionToRead(datasetPath)
    folds = generate_kfolds(df, "target", 8)    
    
    for kIndex in range(8):
        x, y, xEval, yEval = GenerateTrainAndEval(folds, nClass, kIndex)

        nn = NeuralNetwork(alpha)
        nn.readNetworkArchitecture(netWorkPath)

        nn.train(x, y, xEval, yEval, numberIterations, saveResults = "k8Runs/" + netWorkPath.split("/")[1].split(".")[0] + "a" + str(alpha) + ".txt", saveResultsEveryInterval = saveNumber, batchSize=100)

if __name__ == "__main__":
    runTrain(readDataWine, "datasets/wine.data", "testfiles/wineNetwork1.txt", 700, 0.1, 3)
#    runTrain(readDataWine, "datasets/wine.data", "testfiles/wineNetwork1.txt", 700, 0.3, 3)

    runTrain(readDataWine, "datasets/wine.data", "testfiles/wineNetwork2.txt", 700, 0.1, 3)
#    runTrain(readDataWine, "datasets/wine.data", "testfiles/wineNetwork2.txt", 700, 0.3, 3)

    runTrain(readDataWine, "datasets/wine.data", "testfiles/wineNetwork3.txt", 700, 0.1, 3)
#    runTrain(readDataWine, "datasets/wine.data", "testfiles/wineNetwork3.txt", 700, 0.3, 3)

    runTrain(readDataWine, "datasets/wine.data", "testfiles/wineNetwork4.txt", 700, 0.1, 3)


#    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork1.txt", 1300, 0.1, 2)
    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork1.txt", 1200, 0.2, 2)

#    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork2.txt", 1300, 0.1, 2)
    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork2.txt", 1200, 0.2, 2)

#    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork3.txt", 1300, 0.1, 2)
    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork3.txt", 1200, 0.2, 2)

#    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork4.txt", 1300, 0.1, 2)
    runTrain(readDataDiabetes, "datasets/pima.tsv", "testfiles/diabetesNetwork4.txt", 1200, 0.2, 2)



    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork1.txt", 500, 0.1, 2, saveNumber=1)
#    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork1.txt", 500, 0.3, 2, saveNumber=1)

    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork2.txt", 500, 0.1, 2, saveNumber=1)
#    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork2.txt", 500, 0.3, 2, saveNumber=1)

    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork3.txt", 500, 0.1, 2, saveNumber=1)
#    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork3.txt", 500, 0.3, 2, saveNumber=1)

    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork4.txt", 500, 0.1, 2, saveNumber=1)
#    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork4.txt", 500, 0.3, 2, saveNumber=1)

    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork5.txt", 500, 0.1, 2, saveNumber=1)
#    runTrain(readDataIonosphere, "datasets/ionosphere.data", "testfiles/ionosphereNetwork5.txt", 500, 0.3, 2, saveNumber=1)


