import scanpy as sc
import pandas as pd

adata= sc.read_h5ad("HCL_MAGIC_merge_breast_testis_500gene.h5ad")
df = adata.to_df().T

df.to_pickle("HCL_MAGIC_merge_breast_testis_500gene.p")