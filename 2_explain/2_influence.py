import scanpy as sc
import seaborn as sns
import pandas as pd
import numpy as np

influence = 1- pd.read_pickle("./influence_conv1.p", compression= 'xz')
influence.columns = influence.columns.map(lambda x: x.decode())
influence

# Positive influence
influence_pos = influence.copy()
influence_pos[influence_pos<0] = 0

adata_pos = sc.AnnData(influence_pos.T)
adata_pos

sc.pp.normalize_total(adata_pos, target_sum=1)
sc.pp.scale(adata_pos, max_value=1)
sns.clustermap(adata_pos.to_df().T, col_cluster=False, cmap='vlag')


# Negative influence
influence_neg = influence.copy()
influence_neg[influence_neg>0] = 0
influence_neg = influence_neg.abs()

adata_neg = sc.AnnData(influence_neg.T)
adata_neg

sc.pp.normalize_total(adata_neg, target_sum=1)
sc.pp.scale(adata_neg, max_value=1)
sns.clustermap(adata_neg.to_df().T, col_cluster=False, cmap='vlag')