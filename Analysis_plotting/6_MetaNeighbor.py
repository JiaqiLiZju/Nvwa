#!/usr/bin/env python
# coding: utf-8

# # MetaNeighbor analysis
# 
# 
# __Author:__ Jingjing WANG
# 
# __Date:__ March. 2021
# 
# __Two Species:__ Species1 (Mouse) and Species2 (Mouse)


import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pymn
import re
import matplotlib
import random
import bottleneck
from scipy import sparse


get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#These change plot aesthetics
sns.set(style='white', font_scale=1.25)
plt.rc("axes.spines", top=False, right=False)
plt.rc('xtick', bottom=True)
plt.rc('ytick', left=True)

adata_S = sc.read_h5ad('./Zebrafish_MAGIC.h5ad')
meta_S = pd.read_table('./Zebrafish_cellatlas.annotation.20210117.txt',sep="\t",header=0,index_col="Cell")
adata_S
adata_S_df = adata_S.to_df().T
adata_S_df.columns = adata_S_df.columns.astype("str")
adata_S_df.index = adata_S_df.index.astype("str")
cell_index = adata_S_df.columns.intersection(meta_S.index)


random.seed(10)
rc = random.sample(range (0,cell_index.shape[0]),k=int(cell_index.shape[0] * 0.5))
cell_index1 = cell_index[rc]
cell_index2 = cell_index.difference(cell_index1)
adata_S1_df_use = adata_S_df.loc[:,cell_index1]
adata_S2_df_use = adata_S_df.loc[:,cell_index2]
meta_S1 = meta_S.loc[cell_index1,:]
meta_S1[['Sample']] = "Sample1"
meta_S2 = meta_S.loc[cell_index2,:]
meta_S2[['Sample']] = "Sample2"
meta_data = meta_S1.append(meta_S2)
adata_S2_df_use.index = adata_S1_df_use.index
adata_SC1 = sc.AnnData(adata_S1_df_use.T)
adata_SC2 = sc.AnnData(adata_S2_df_use.T)


# ##  Step 2 MetaNeighbor: Hierarchical cell type replicability analysis
adata_reference = adata_SC1.concatenate(adata_SC2)
adata_reference.obs_names = meta_data.index

adata_reference

adata_reference.obs['Celltype'] = meta_data['Celltype']
adata_reference.obs['Cellcluster'] = meta_data['Cellcluster']
adata_reference.obs['study_id'] = meta_data['Sample']

adata_reference
adata_reference.obs['study_id']


def compute_var_genes(adata, return_vect=True):
    """Compute variable genes for an indiviudal dataset


    Arguments:
        adata {[type]} -- AnnData object containing a signle dataset

    Keyword Arguments:
        return_vect {bool} -- Boolean to store as adata.var['higly_variance']
            or return vector of booleans for varianble gene membership (default: {False})

    Returns:
        np.ndarray -- None if saving in adata.var['highly_variable'], array of booleans if returning of length ngenes
    """

    if sparse.issparse(adata.X):
        median = csc_median_axis_0(sparse.csc_matrix(adata.X))
    else:
        median = bottleneck.median(adata.X, axis=0)
    variance = np.var(adata.X, axis=0) ##.A
    bins = np.quantile(median, q=np.linspace(0, 1, 11), interpolation="midpoint")
    digits = np.digitize(median, bins, right=True)

    selected_genes = np.zeros_like(digits)
    for i in np.unique(digits):
        filt = digits == i
        var_tmp = variance[filt]
        bins_tmp = np.nanquantile(var_tmp, q=np.linspace(0, 1, 5))
        g = np.digitize(var_tmp, bins_tmp)
        selected_genes[filt] = (g >= 4).astype(float)

    if return_vect:
        return selected_genes.astype(bool)
    else:
        adata.var["highly_variable"] = selected_genes.astype(bool)

def variableGenes_use(adata, study_col, return_vect=False):
    """Comptue variable genes across data sets

    Identifies genes with high variance compared to their median expression
    (top quartile) within each experimentCertain function

    Arguments:
        adata {AnnData} -- AnnData object containing all the single cell experiements concatenated together
        study_col {str} -- String referencing column in andata.obs that identifies study label for datasets

    Keyword Arguments:
        return_vect {bool} -- Boolean to store as adata.var['higly_variance']
            or return vector of booleans for varianble gene membership (default: {False})

    Returns:
        np.ndarray -- None if saving in adata.var['highly_variable'], array of booleans if returning of length ngenes
    """

    assert study_col in adata.obs_keys(), "Study Col not in obs data"

    studies = np.unique(adata.obs[study_col])
    genes = adata.var_names
    var_genes_mat = pd.DataFrame(index=genes)

    for study in studies:
        slicer = adata.obs[study_col] == study
        genes_vec = compute_var_genes(adata[slicer])
        var_genes_mat.loc[:, study] = genes_vec.astype(bool)
    var_genes = np.all(var_genes_mat, axis=1)
    if return_vect:
        return var_genes
    else:
        adata.var["highly_variable"] = var_genes


def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]
def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.
    This function is used to support sparse matrices; it modifies data in-place
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    middle, is_odd = divmod(n_elems, 2)
    data.sort()

    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)

    return (
        _get_elem_at_rank(middle - 1, data, n_negative, n_zeros)
        + _get_elem_at_rank(middle, data, n_negative, n_zeros)
    ) / 2.0


def csc_median_axis_0(X):
    """Find the median across axis 0 of a CSC matrix.
    It is equivalent to doing np.median(X, axis=0).
    Parameters
    ----------
    X : CSC sparse matrix, shape (n_samples, n_features)
        Input data.
    Returns
    -------
    median : ndarray, shape (n_features,)
        Median.
    """
    if not isinstance(X, sparse.csc_matrix):
        raise TypeError("Expected matrix of CSC format, got %s" % X.format)

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):

        # Prevent modifying X in place
        data = np.copy(X.data[start:end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)

    return median


adata_reference
variableGenes_use(adata_reference, study_col='study_id')
HVG = adata_reference.var['highly_variable']
HVG[HVG].index
adata_reference
pymn.MetaNeighborUS(adata_reference,
                    study_col='study_id',
                    ct_col='Celltype',
                    fast_version=True)


pymn.plotMetaNeighborUS(adata_reference, figsize=(10, 10), cmap='coolwarm', fontsize=10)
pymn.topHits(adata_reference, threshold=0.9)
adata_reference.uns['MetaNeighborUS']
adata_reference.uns['MetaNeighborUS_topHits']

pymn.MetaNeighborUS(adata_reference,
                    study_col='study_id',
                    ct_col='Celltype',
                    fast_version=True,
                    symmetric_output=False,
                    one_vs_best=True)


pymn.plotMetaNeighborUS(adata_reference,
                        cmap='coolwarm',
                        figsize=(10, 10),
                        mn_key='MetaNeighborUS_1v1',
                        xticklabels=True,
                        yticklabels=True,
                        fontsize=7)


adata_reference
adata_reference.var['highly_variable'].to_csv("HVGS.csv")
adata_reference.uns['MetaNeighborUS'].to_csv("AUROC.csv")
adata_reference.uns['MetaNeighborUS_topHits'].to_csv("tophit_0.9.csv")
adata_reference.uns['MetaNeighborUS_1v1'].to_csv("AUROC_best_hits.csv")
adata_reference.write("adata_reference.h5ad")
adata_reference.obs.to_csv("adata_reference.obs.csv")
import time
from datetime import datetime
now=datetime.now()
print(now)
