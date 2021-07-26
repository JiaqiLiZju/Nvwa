import pandas as pd
import numpy as np

from matplotlib_venn import *
import matplotlib.pyplot as plt

fname1 = "Predict_Neuron_Mouse12.Plus.bedGraph"
df1 = pd.read_csv(fname1, sep='\t', header=None)
df1.columns = ["Chr", "Start", "End", "Value"]
df1["Peak"] = 0
df1["Pos"] = ((df1.Start + df1.End) / 2).astype(int)

df1 = df1[abs(df1.Value - 0.451436) > 0.001]

t1 = np.quantile(df1.Value, 0.9)
df1.loc[df1.Value > t1, "Peak"] = 1

df1 = df1[df1["Peak"] == 1]



fname2 = "Ex_neurons_CPN-clusters_5-cluster_1.mm10.BedGraph"
df2 = pd.read_csv(fname2, sep='\t', header=None)
df2.columns = ["Chr", "Start", "End", "Value"]
df2 = df2[df2.Chr == "chr8"]
df2["Peak"] = 0
df2["Pos"] = ((df2.Start + df2.End) / 2).astype(int)

t2 = np.quantile(df2.Value, 0.9)
df2.loc[df2.Value > t2, "Peak"] = 1

df2 = df2[df2["Peak"] == 1]



for bin_size in [5000, 10000, 20000]:

    chr8 = pd.DataFrame({"Chr":"chr8", "Pos":range(0, 129401213, bin_size)})

    chr8[["Peak1", "Peak2"]] = 0

    for pos in df1.Pos.values:
        idx = int(pos / bin_size)
        chr8.iloc[idx:idx+2, 2] = 1
        
    chr8.Peak1.unique()

    for pos in df2.Pos.values:
        idx = int(pos / bin_size)
        chr8.iloc[idx:idx+2, 3] = 1
        
    chr8.Peak2.unique()

    chr8 = chr8[(chr8.Peak1==1)|(chr8.Peak2==1)]
    chr8.to_csv("./bin_size_"+str(bin_size)+".csv")

    subsets = [set(chr8[chr8.Peak1==1].Pos), set(chr8[chr8.Peak2==1].Pos)]

    plt.figure()
    venn2(subsets=subsets, set_labels = ('Pred', 'ATAC'))
    plt.savefig("./bin_size_"+str(bin_size)+".pdf")
    plt.show()
    plt.close()