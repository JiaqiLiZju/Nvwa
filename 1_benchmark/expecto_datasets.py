import os, gc, h5py, pickle
from sys import argv
import numpy as np
import pandas as pd

nvwa_dataset_fname, Xreducedall_fname, Xreducedall_geneanno_fname, output_fname = argv[1:]
# "../nvwa-pse-official/data-hcl/HCL_pseudocell_bin.h5", \
# "./resources/Xreducedall.2002.npy", \
# "./resources/geneanno.csv", \
# "HCL_pseudocell_bin_Expecto.h5"

# load nvwa-datasets
h5file = h5py.File(nvwa_dataset_fname, 'r')
celltype = h5file["celltype"][:]
train_gene_nvwa = h5file["train_gene"][:]
test_gene_nvwa = h5file["test_gene"][:]
val_gene_nvwa = h5file["val_gene"][:]
# train_data = h5file["train_data"][:].astype(np.float32)
y_train_nvwa = h5file["train_label"][:].astype(np.float32)
# val_data = h5file["val_data"][:].astype(np.float32)
y_val_nvwa = h5file["val_label"][:].astype(np.float32)
# test_data = h5file["test_data"][:].astype(np.float32)
y_test_nvwa = h5file["test_label"][:].astype(np.float32)
h5file.close()

# load expecto_Xreduced
Xreducedall = np.load(Xreducedall_fname)
geneanno = pd.read_csv(Xreducedall_geneanno_fname)
gene_expecto = geneanno.symbol.values

# filter 
filt = np.asarray(geneanno.iloc[:, -1] != 'rRNA')
Xreducedall = Xreducedall[filt]
gene_expecto = gene_expecto[filt]
print(Xreducedall.shape)
print(gene_expecto.shape)

# get idx
train_idx, test_idx, val_idx = [], [], []
for gene in gene_expecto:
    train_idx.append(gene in train_gene_nvwa)
    test_idx.append(gene in test_gene_nvwa)
    val_idx.append(gene in val_gene_nvwa)

# x
x_train, x_test, x_val = Xreducedall[train_idx], Xreducedall[test_idx], Xreducedall[val_idx]
# gene names
train_gene, test_gene, val_gene = gene_expecto[train_idx], gene_expecto[test_idx], gene_expecto[val_idx]
# y
train_df = pd.DataFrame(y_train_nvwa, index=train_gene_nvwa)
y_train = train_df.loc[train_gene].values
test_df = pd.DataFrame(y_test_nvwa, index=test_gene_nvwa)
y_test = test_df.loc[test_gene].values
val_df = pd.DataFrame(y_val_nvwa, index=val_gene_nvwa)
y_val = val_df.loc[val_gene].values

# check
print(x_train.shape, x_test.shape, x_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)
print(train_gene.shape, test_gene.shape, val_gene.shape)
print("checked, saving...")

# save datasets
compress_args = {'compression': 'gzip', 'compression_opts': 1}

h5file = h5py.File(output_fname, 'w')
h5file.create_dataset("celltype", data=celltype, dtype=h5py.special_dtype(vlen=str))
h5file.create_dataset("train_gene", data=train_gene, dtype=h5py.special_dtype(vlen=str), **compress_args)
h5file.create_dataset("val_gene", data=val_gene, dtype=h5py.special_dtype(vlen=str), **compress_args)
h5file.create_dataset("test_gene", data=test_gene, dtype=h5py.special_dtype(vlen=str), **compress_args)

h5file.create_dataset("train_data", data=x_train, dtype=float, **compress_args)
h5file.create_dataset("train_label", data=y_train, dtype=float, **compress_args)

h5file.create_dataset("val_data", data=x_val, dtype=float, **compress_args)
h5file.create_dataset("val_label", data=y_val, dtype=float, **compress_args)

h5file.create_dataset("test_data", data=x_test, dtype=float, **compress_args)
h5file.create_dataset("test_label", data=y_test, dtype=float, **compress_args)

h5file.close()