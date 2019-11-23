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

    x = df.drop(["target"], axis=1)

    return(x, dfTarget)

if __name__ == "__main__":
    print(generate_kfolds(readDataWine("datasets/wine.data"), "target", 8))

