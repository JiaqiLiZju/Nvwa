import numpy as np
import scanpy as sc
import pandas as pd
import os
import pandas as pd
import gc
import dask.dataframe

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=300,dpi_save=600, facecolor='white')

adata=sc.read('Zebrafish.dge.h5ad')
sc.pp.filter_genes(adata, min_cells=20)
sc.pp.filter_cells(adata, min_genes=0)
adata.obs['n_counts'] = adata.X.sum(axis=1)
sc.pl.violin(adata, ['n_genes', 'n_counts'],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(adata, x='n_counts', y='n_genes')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.001, max_mean=15, min_disp=0.45)
sc.pl.highly_variable_genes(adata)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack',n_comps=100)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.tsne(adata,use_fast_tsne=True,n_jobs=20,perplexity=50,n_pcs=50)
sc.tl.leiden(adata, resolution=2)
sc.pl.tsne(adata, color=['leiden'],palette="Paired",legend_fontsize=5,legend_loc='on data') #add_outline=True,
sc.tl.rank_genes_groups(adata,'leiden',method='wilcoxon')
sc.pl.rank_genes_groups(adata,n_genes=25,sharey=False)
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'logfoldchanges','scores', 'pvals', 'pvals_adj']}).to_csv("markers_wilcoxon.csv")
