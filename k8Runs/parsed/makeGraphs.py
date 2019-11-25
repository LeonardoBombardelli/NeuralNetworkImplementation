import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import pandas as pd

from glob import glob

def createTitle(nameFile):
    return(nameFile.split("/")[-1].replace("a0.2", "").replace("a0.1", "").replace("a0", "").replace(".txt",""))

allFiles = glob("./k8Runs/*.txt")

for name in allFiles:
    df = pd.read_csv(name, names=["Iteration","Accuracy","J","F1"])

    # ax = sns.lineplot(x="Iteration", y="Accuracy", data=df).set_title(createTitle(name))
    # ax = sns.lineplot(x="Iteration", y="J", data=df, color="red").set_title(createTitle(name))
    ax = sns.lineplot(x="Iteration", y="F1", data=df, color="green").set_title(createTitle(name))

    plt.ylim(0, 1)

    # plt.show()
    plt.savefig("./k8Runs/plots/"+createTitle(name)+"F1.png")
    plt.close()