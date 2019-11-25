import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import pandas as pd

from glob import glob

allMeans = glob("./k8Runs/parsed/mean-*.csv")
allStd = glob("./k8Runs/parsed/std-*.csv")

allMeans.sort()
allStd.sort()

df = pd.read_csv(allMeans[0])
df.insert(3, "std", pd.read_csv(allStd[0])["Accuracy"], True)
print(df)
ax = sns.lineplot(x="Iteration", y="Accuracy", data=df)

plt.show()
