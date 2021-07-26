from sys import argv
import numpy as np
import pandas as pd
import h5py, pickle

nvwa_dataset_fname, orth_fname, output_fname = argv[1:]

orth_species = ["brenneri", "briggsae", "latens", "nigoni"]#, "remanei"]

data_type = float

# load nvwa-datasets
h5file = h5py.File(nvwa_dataset_fname, 'r')
celltype = h5file["celltype"][:]
anno = h5file["annotation"][:]
train_gene_nvwa = h5file["train_gene"][:]
test_gene_nvwa = h5file["test_gene"][:]
val_gene_nvwa = h5file["val_gene"][:]
train_data = h5file["train_data"][:]
y_train_nvwa = h5file["train_label"][:]
val_data = h5file["val_data"][:]
y_val_nvwa = h5file["val_label"][:]
test_data = h5file["test_data"][:]
y_test_nvwa = h5file["test_label"][:]
h5file.close()

# orth
df = pd.read_csv(orth_fname)

x_train = [train_data]
y_train = [y_train_nvwa]
train_gene = [train_gene_nvwa]

x_val = [val_data]
y_val = [y_val_nvwa]
val_gene = [val_gene_nvwa]

for specie in orth_species:
    print("processing %s..." % specie)
    # load dataset
    genome = "/share/home/guoguoji/NvWA/Dataset/Celegan/onehot/c." + specie + ".onehot.p"
    data = pickle.load(open(genome,'rb'))

    orth = {k:v for k,v in zip(df.Celegans_ID, df["C" + specie + "_id"])}

    # train set
    x, y, g = [], [], []
    for idx, gene in enumerate(train_gene_nvwa):
        if gene in orth.keys():
            gene_orth = orth[gene]
            seq = data[gene_orth][-1]
            if seq.shape == (19999, 4):
                seq = np.vstack((seq, np.zeros((1,4))))
                print(gene_orth)
            if seq.shape == (20000, 4):
                g.append(gene_orth)
                x.append(seq)
                y.append(y_train_nvwa[idx, :])

    x = np.array(x, dtype=bool).swapaxes(1, -1)
    y = np.array(y, dtype=data_type)
    x_train.append(x)
    y_train.append(y)
    train_gene.append(g)

    # val set
    x, y, g = [], [], []
    for idx, gene in enumerate(val_gene_nvwa):
        if gene in orth.keys():
            gene_orth = orth[gene]
            seq = data[gene_orth][-1]
            if seq.shape == (19999, 4):
                seq = np.vstack((seq, np.zeros((1,4))))
                print(gene_orth)
            if seq.shape == (20000, 4):            
                g.append(gene_orth)
                x.append(seq)
                y.append(y_val_nvwa[idx, :])
    x = np.array(x, dtype=bool).swapaxes(1, -1)
    y = np.array(y, dtype=data_type)
    x_train.append(x)
    y_train.append(y)
    val_gene.append(g)

    # train In Test
    x, y, g = [], [], []
    for idx, gene in enumerate(test_gene_nvwa):
        if gene in orth.keys():
            gene_orth = orth[gene]
            seq = data[gene_orth][-1]
            if seq.shape == (19999, 4):
                seq = np.vstack((seq, np.zeros((1,4))))
                print(gene_orth)
            if seq.shape == (20000, 4):            
                g.append(gene_orth)
                x.append(seq)
                y.append(y_test_nvwa[idx, :])
    x = np.array(x, dtype=bool).swapaxes(1, -1)
    y = np.array(y, dtype=data_type)
    x_train.append(x)
    y_train.append(y)
    train_gene.append(g)

# train set
x_train = np.vstack(x_train)
y_train = np.vstack(y_train)
train_gene = np.hstack(train_gene)
print(x_train.shape)

# val set
x_val = np.vstack(x_val)
y_val = np.vstack(y_val)
val_gene = np.hstack(val_gene)
print(x_val.shape)

# test set
test_gene = test_gene_nvwa
x_test = test_data
y_test = y_test_nvwa

# save datasets
compress_args = {'compression': 'gzip', 'compression_opts': 1}

h5file = h5py.File(output_fname, 'w')
h5file.create_dataset("celltype", data=celltype, dtype=h5py.special_dtype(vlen=str))
h5file.create_dataset("annotation", data=anno, dtype=h5py.special_dtype(vlen=str), **compress_args)

h5file.create_dataset("train_gene", data=train_gene, dtype=h5py.special_dtype(vlen=str), **compress_args)
h5file.create_dataset("val_gene", data=val_gene, dtype=h5py.special_dtype(vlen=str), **compress_args)
h5file.create_dataset("test_gene", data=test_gene, dtype=h5py.special_dtype(vlen=str), **compress_args)

h5file.create_dataset("train_data", data=x_train, dtype=bool, **compress_args)
h5file.create_dataset("train_label", data=y_train, dtype=data_type, **compress_args)

h5file.create_dataset("val_data", data=x_val, dtype=bool, **compress_args)
h5file.create_dataset("val_label", data=y_val, dtype=data_type, **compress_args)

h5file.create_dataset("test_data", data=x_test, dtype=bool, **compress_args)
h5file.create_dataset("test_label", data=y_test, dtype=data_type, **compress_args)

h5file.close()