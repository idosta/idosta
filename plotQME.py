import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('29112021.csv')
I = np.zeros((df.loc[df["epsilon"] == 0].shape[0], df.loc[df["v"] == 0].shape[0]))
n = 0
x = []
y = []
for i in df["v"].drop_duplicates():
    dfv = df.loc[df["v"] == i]
    x.append(i)
    print(dfv.shape[0])
    m = 0
    for j in dfv["epsilon"]:
        dfveps = dfv.loc[dfv["epsilon"] == j]
        I[n, m] = dfveps["I"].values
        m = m + 1
        if i == 0:
            y.append(j)
    n = n + 1
print(I)
plt.matshow(I)
plt.colorbar()
plt.show()
