import magic
import numpy as np
import scanpy as sc
import pandas as pd
import scprep
import scipy

adata = sc.read("Zebrafish.dge.h5ad")

######### preprocess
sc.pp.filter_cells(adata, min_genes = 600)
sc.pp.normalize_total(adata, target_sum = 1e5)
print(adata)
sc.pp.log1p(adata, base = 2)
df_adata = pd.DataFrame(adata.X, columns = adata.var_names, index = adata.obs_names)

######### MAGIC
magic_op = magic.MAGIC()
df_adata_magic = magic_op.fit_transform(df_adata_log)

df_adata_magic = df_adata_magic * 100
df_adata_magic = df_adata_magic.astype(int)
df_adata_magic = df_adata_magic/100

adata_magic = sc.AnnData(df_adata_magic)
adata_magic.write("Zebrafish_MAGIC.h5ad", compression = True)




