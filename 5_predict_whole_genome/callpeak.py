import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sys import argv

fname1, null_path = argv[1:] # "./predicted_tracks/Predict_Neuron_Mouse12.Mean.bedGraph"
os.makedirs("./Peak_tracks/", exist_ok=True)
celltype, strand = fname1.split('/')[-1].replace("Predict_", "").split('.')[:2]
output_prefix = "./Peak_tracks/" + "Peak_" + celltype + "." + strand + "."

df1 = pd.read_csv(fname1, sep='\t', header=None)
df1.columns = ["Chr", "Start", "End", "Value"]
df1["Pos"] = ((df1.Start + df1.End) / 2).astype(int)

x = df1.Value.values
t1 = np.quantile(x, 0.99)
print(t1)

df1["Peak_quantile"] = 0
df1.loc[df1.Value > t1, "Peak_quantile"] = 1
df1.loc[df1.Peak_quantile==1, ["Chr", "Start", "End", "Value"]].to_csv(output_prefix+"quantile.bed", sep='\t', header=False, index=False)

# null_random_background
fname_null = os.path.join(null_path, "null_random_background.csv")
df_null = pd.read_csv(fname_null, sep=',', header=0, index_col=0)
null = df_null[celltype].values
print(null.shape)

P = []
for idx in range(x.shape[0]):
    w, p = wilcoxon(np.repeat(x[idx], null.shape[0]), y=null, alternative='greater')
    if p < 0.05:
        print(idx, w, p)
    P.append(p)
    
peak_mask = np.array(P) < 0.05
peak_sum = np.sum(peak_mask) 

df1["Peak_null_promoter"] = 0
df1["P_null_promoter"] = P
df1.loc[peak_mask, "Peak_null_promoter"] = 1
df1.loc[df1.Peak_null_promoter==1, ["Chr","Start","End", "P_null_promoter"]].to_csv(output_prefix+"peak_null_promoter.bed", sep='\t', header=False, index=False)

# null preds_random_location
fname_null = os.path.join(null_path, "preds_random_location_Mean.csv")
df_null = pd.read_csv(fname_null, sep=',', header=0, index_col=0)
null = df_null[celltype].values

P = []
for idx in range(x.shape[0]):
    w, p = wilcoxon(np.repeat(x[idx], null.shape[0]), y=null, alternative='greater')
    if p < 0.01:
        print(idx, w, p)
    P.append(p)

peak_mask = np.array(P) < 0.05
peak_sum = np.sum(peak_mask) 

df1["Peak_random_location"] = 0
df1["P_random_location"] = P
df1.loc[peak_mask, "Peak_random_location"] = 1
df1.loc[df1.Peak_random_location==1, ["Chr","Start","End","P_random_location"]].to_csv(output_prefix+"peak_random_location.bed", sep='\t', header=False, index=False)

df1.to_csv(output_prefix+"peak_all.bed", sep='\t', header=False, index=False)
