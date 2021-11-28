import os, glob
import pandas as pd
import numpy as np

motif_cnt_d = {}
for path in glob.glob("../CrossValid/tomtom_vs*"):
    df = pd.read_csv(os.path.join(path, "tomtom.tsv"), sep='\t', header=0, skipfooter=3)
    df = df[df['q-value'] < 0.05]
    motif_cnt = df.groupby("Query_ID")["Target_ID"].count()
    motif_cnt_d[path] = motif_cnt

motif_cnt = pd.DataFrame(motif_cnt_d).fillna(0).astype(int)
sum_cnt = motif_cnt.sum(1)
sum_match = np.count_nonzero(motif_cnt, 1)

motif_cnt['sum_cnt'] = sum_cnt
motif_cnt['sum_match'] = sum_match
motif_cnt.to_csv("motif_reproduce_inCV.csv")