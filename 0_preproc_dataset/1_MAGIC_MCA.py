import magic
import numpy as np
import scanpy as sc
import pandas as pd
import os

adata = sc.read("../EarthWorm_normalize_merge_20210915_mincell1.h5ad")
print(adata)

sc.pp.normalize_total(adata, target_sum = 1e5)
sc.pp.filter_genes(adata, min_cells = 20)
# sc.pp.filter_cells(adata, min_genes = 50)
print(adata)

sc.pp.log1p(adata, base = 2)
df_adata = pd.DataFrame(adata.X, columns = adata.var_names, index = adata.obs_names)

magic_op = magic.MAGIC()
df_adata_magic = magic_op.fit_transform(df_adata)

#adata_magic = sc.AnnData(df_adata_magic)
#adata_magic.write("MCA_MAGIC_merge_E95_E105_raw.h5ad", compression = True)
#adata_temp = sc.read("./MCA_MAGIC_merge_E95_E105_raw.h5ad")
#df_adata_magic = pd.DataFrame(adata_temp.X, columns = adata_temp.var_names, index = adata_temp.obs_names)

df_adata_magic2 = df_adata_magic*100
df_adata_magic2 = df_adata_magic2.astype(int)
df_adata_magic2 = df_adata_magic2/100

#df2csv(df_adata_magic2,"HCL_MAGIC_int.csv")
adata_magic2 = sc.AnnData(df_adata_magic2)
adata_magic2.write("EarthWorm_mincell20_MAGIC.h5ad", compression = True)

df = adata_magic2.to_df().T
df[df >= 0.001] = 1
df[df < 0.001] = 0
df.to_pickle("EarthWorm_mincell20_t1e-3.p")
print(df.sum(0))
print(df.sum(0)/df.shape[0])

import magic
import numpy as np
import scanpy as sc
import pandas as pd
import os

adata = sc.read("../EarthWorm_normalize_merge_20210915_mincell1.h5ad")
print(adata)
sc.pp.filter_genes(adata, min_cells = 10)
print(adata)

df = adata.to_df().T
df[df >= 0.001] = 1
df[df < 0.001] = 0
df.to_pickle("EarthWorm_mincell20_t1e-3.p")
print(df.sum(0))
print(df.sum(0)/df.shape[0])
