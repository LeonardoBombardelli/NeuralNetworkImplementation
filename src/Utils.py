import pandas as pd
import numpy as np


def calculateF1(results: dict):
    sum  = 0.0
    size = 0.0

    for c in results:
        if not results[c]['eval']:
            continue
        prec = 1.0
        rev  = 1.0

        if results[c]['VP'] != 0 or results[c]['FP'] != 0:
            prec = float(results[c]['VP']) / float(results[c]['VP'] + results[c]['FP'])

        if results[c]['VP'] != 0 and results[c]['FN'] != 0:
            rev  = float(results[c]['VP']) / float(results[c]['VP'] + results[c]['FN'])

        sum += 2.0 * ((prec * rev) / (prec + rev))
        size += 1.0

    return sum / size

def bootstrap(dataset: pd.DataFrame, n: int):
    trainData = dataset.sample(n, replace=True)
    testData  = dataset.loc[~dataset.index.isin(trainData.index)]
    return trainData.reset_index(drop=True), testData.reset_index(drop=True)

def generate_kfolds(dataset: pd.DataFrame, target: str, k: int):
    N = int(len(dataset) / k)
    folds = []

    for i in range(k-1):
        datasetLen = len(dataset)
        df = dataset.sample(1).groupby(target, group_keys=False, sort=False).apply(
             lambda x: x.sample(int(len(x)*N/len(dataset))))

        dataset = dataset.loc[~dataset.index.isin(df.index)]

        samples_left = int(N - len(df))
        if samples_left > 0:
            df = pd.concat([df, dataset.sample(samples_left)])

        dataset = dataset.loc[~dataset.index.isin(df.index)]
        folds.append(df.reset_index(drop=True))

    folds.append(dataset.reset_index(drop=True))
    return folds

def getSeparator(path: str):
    line = ''

    with open(path) as file:
        line = next(file)

    if line.find(',') != -1:
        return ','
    else:
        return ';'

def readDataWine(datawinePath: str):
    df = pd.read_csv(datawinePath)

    dfTarget = df["target"]
    df = (df-df.min())/(df.max()-df.min())
    df["target"] = dfTarget

#    x = df.drop(["target"], axis=1)

#    dfTarget = OneHotEncoding(dfTarget, 3)

    return(df)

def readDataDiabetes(datadiabetesPath):
    df = pd.read_csv(datadiabetesPath, sep="\t")

    dfTarget = df["target"]
    df = (df-df.min())/(df.max()-df.min())
    df["target"] = dfTarget

#    x = df.drop(["target"], axis=1)

#    dfTarget = OneHotEncoding(dfTarget, 2)

    return(df)

def readDataIonosphere(dataionospherePath):
    df = pd.read_csv(dataionospherePath)

    cleanup = {
        '33': {'b': 1, 'g': 0}
    }

    df.replace(cleanup, inplace=True)

    dfTarget = df["target"]
    df = (df-df.min())/(df.max()-df.min())
    df["target"] = dfTarget

#    x = df.drop(["target", "0"], axis=1)

#    dfTarget = OneHotEncoding(dfTarget, 2)

    df = df.drop(["0"], axis=1)

    return(df)

def OneHotEncoding(listToEncode, numberOfClasses: int):
    ys = []
    for i in listToEncode.tolist():
        yl = [0] * numberOfClasses
        yl[i-1] = 1
        ys.append(yl)
    return(ys)


def DecodeOneHot(decodeList):
    if(isinstance(decodeList, list)):
        return(decodeList.index(max(decodeList)))
    else:
        return(np.argmax(decodeList))

def GenerateTrainAndEval(folds, nClass, kIndex):
        foldTrain = []
        for i in range(8):
            if i != kIndex:
                foldTrain.append(folds[i])

        foldTrain = pd.concat(folds[1:])

        y = OneHotEncoding(foldTrain["target"], nClass)
        x = foldTrain.drop(["target"], axis=1)
        x = x.values.tolist()

        foldEval = folds[kIndex]

        yEval = OneHotEncoding(foldEval["target"], nClass)
        xEval = foldEval.drop(["target"], axis=1)
        xEval = xEval.values.tolist()

        return(x, y, xEval, yEval)

# if __name__ == "__main__":
#     print(generate_kfolds(readDataWine("datasets/wine.data"), "target", 8))
