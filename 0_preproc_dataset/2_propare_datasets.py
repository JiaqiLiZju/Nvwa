import os, gc, logging
import h5py, pickle
from sys import argv

import numpy as np
import pandas as pd
# import datatable

from sklearn.model_selection import KFold, train_test_split

mode, genome, label, annotation, output_dir = argv[1:6]

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log.' + mode + '.txt',
                    filemode='w')

logging.info("######### args ##########")
for arg in argv[1:]:
    logging.info(arg)

# set random seed
random_seed = 77

data_type = np.float32 #bool
compress_args = {'compression': 'gzip', 'compression_opts': 1}

# load annotation
anno = pd.read_csv(annotation, sep='\t', header=0, encoding="unicode_escape")
anno.columns = ["Cell", "Species", "Celltype", "Cluster"]
anno.index = anno.Cell
logging.info(anno.shape)

# load labels
# cell_lineage = datatable.fread(label, sep=',', header=True).to_pandas()
# cell_lineage = cell_lineage.set_index("C0")
# cell_lineage.columns = cell_lineage.columns.map(lambda x: x.split("'")[1])
# cell_lineage = pd.read_csv(label, sep=',', header=0, index_col=0, engine='c').fillna(0)
cell_lineage = pd.read_pickle(label)
cell_lineage = cell_lineage.astype(data_type)
logging.info(cell_lineage.shape)

# set intersection
select = anno.index.intersection(cell_lineage.columns)
anno = anno.loc[select]

# sort
anno = anno.sort_values(["Cluster","Celltype","Cell"])
celltypes = anno.Cell.values
anno = anno.values
logging.info(anno.shape)

cell_lineage = cell_lineage[celltypes]
logging.info(cell_lineage.shape)

# load dataset
data = pickle.load(open(genome,'rb'))
logging.info(len(data))

# X, y, geneName = [], [], []
# for gene in data.keys():
#     gene_HCL = gene
#     # gene_HCL = gene.replace("NEMVEDRAFT_", '')
#     # if gene in genename_d:
#     #     gene_HCL =  genename_d[gene]
#     # else:
#     #     continue
#     if gene_HCL in cell_lineage.index and not gene_HCL.startswith('MT'):
#         label = cell_lineage.loc[gene_HCL].values
#         seq = data[gene][-1]
#         # if type(label)==np.int64 and seq.dtype == np.int and seq.shape == (20000, 4):
#         if seq.shape == (20000, 4):
#             # seq = seq[3500:13500,:]
#             X.append(seq)
#             y.append(label)
#             geneName.append(gene)

X, y, geneName = [], [], []
for gene in cell_lineage.index.values:
    label = cell_lineage.loc[gene].values
    if gene not in data.keys():
        logging.info("Not in Genome:\t%s" % gene)
        gene = gene.replace(".", "-", 1)
    if gene in data.keys():
        seq = data[gene][-1]
        if seq.shape == (19999, 4):
            seq = np.vstack((seq, np.zeros((1,4))))
        if seq.shape == (20000, 4):            
            X.append(seq)
            y.append(label)
            geneName.append(gene)
    else:
        logging.info("Still Not in Dataset:\t%s" % gene)

geneName = np.array([gene.encode() for gene in geneName]) # encode for h5py
X = np.array(X, dtype=bool).swapaxes(1, -1) # N*4*20000
y = np.array(y, dtype=data_type)

logging.info(X.shape)
logging.info(y.shape)

del data, cell_lineage
gc.collect()

if mode == "train_test_split":
    # train and test split
    # test_size > validation_size and use validation to optimize the epoch and lr used
    x_train_idx, x_test_idx, y_train, y_test = train_test_split(range(len(X)), y, test_size=1000, shuffle=True, random_state=random_seed)
    x_train_val = X[x_train_idx]; x_test = X[x_test_idx]
    gene_train_val = geneName[x_train_idx]; gene_test = geneName[x_test_idx]

    x_train_idx, x_val_idx, y_train, y_val = train_test_split(range(len(x_train_val)), y_train, test_size=1000, shuffle=True, random_state=random_seed)
    x_train = x_train_val[x_train_idx]; x_val = x_train_val[x_val_idx]
    gene_train = gene_train_val[x_train_idx]; gene_val = gene_train_val[x_val_idx]

    logging.info(x_train.shape)
    logging.info(x_test.shape)
    logging.info(x_val.shape)

    h5file = h5py.File(output_dir, 'w')
    h5file.create_dataset("celltype", data=celltypes, dtype=h5py.special_dtype(vlen=str))
    h5file.create_dataset("annotation", data=anno, dtype=h5py.special_dtype(vlen=str), **compress_args)

    h5file.create_dataset("train_gene", data=gene_train, dtype=h5py.special_dtype(vlen=str), **compress_args)
    h5file.create_dataset("val_gene", data=gene_val, dtype=h5py.special_dtype(vlen=str), **compress_args)
    h5file.create_dataset("test_gene", data=gene_test, dtype=h5py.special_dtype(vlen=str), **compress_args)

    h5file.create_dataset("train_data", data=x_train, dtype=bool, **compress_args)
    h5file.create_dataset("train_label", data=y_train, dtype=data_type, **compress_args)

    h5file.create_dataset("val_data", data=x_val, dtype=bool, **compress_args)
    h5file.create_dataset("val_label", data=y_val, dtype=data_type, **compress_args)

    h5file.create_dataset("test_data", data=x_test, dtype=bool, **compress_args)
    h5file.create_dataset("test_label", data=y_test, dtype=data_type, **compress_args)

    h5file.close()

if mode == "cross_valid":
    # train and test split
    x_train_idx, x_test_idx, y_train_val, y_test = train_test_split(range(len(X)), y, test_size=1000, shuffle=True, random_state=random_seed)
    x_train_val = X[x_train_idx]; x_test = X[x_test_idx]
    gene_train_val = geneName[x_train_idx]; gene_test = geneName[x_test_idx]

    kf = KFold(n_splits=10, random_state=random_seed, shuffle=True)
    fold = 0 
    for train_idx, val_idx in kf.split(range(len(x_train_val))):
        fold += 1
        logging.info("processing fold %d" % fold)

        x_train, x_val = x_train_val[train_idx], x_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        gene_train, gene_val = gene_train_val[train_idx], gene_train_val[val_idx]

        logging.info(x_train.shape)
        logging.info(x_test.shape)
        logging.info(x_val.shape)

        output_fname = os.path.join(output_dir, "Dataset.cross_valid_"+str(fold)+".h5")
        h5file = h5py.File(output_fname, 'w')
        h5file.create_dataset("celltype", data=celltypes, dtype=h5py.special_dtype(vlen=str))
        h5file.create_dataset("annotation", data=anno, dtype=h5py.special_dtype(vlen=str), **compress_args)
        
        h5file.create_dataset("train_gene", data=gene_train, dtype=h5py.special_dtype(vlen=str), **compress_args)
        h5file.create_dataset("val_gene", data=gene_val, dtype=h5py.special_dtype(vlen=str), **compress_args)
        h5file.create_dataset("test_gene", data=gene_test, dtype=h5py.special_dtype(vlen=str), **compress_args)
        
        h5file.create_dataset("train_data", data=x_train, dtype=bool, **compress_args)
        h5file.create_dataset("train_label", data=y_train, dtype=data_type, **compress_args)

        h5file.create_dataset("val_data", data=x_val, dtype=bool, **compress_args)
        h5file.create_dataset("val_label", data=y_val, dtype=data_type, **compress_args)

        h5file.create_dataset("test_data", data=x_test, dtype=bool, **compress_args)
        h5file.create_dataset("test_label", data=y_test, dtype=data_type, **compress_args)

        h5file.close()

if mode == "leave_chrom":

    gtf_fname, chrom = argv[-2:]
    chrom = str(chrom)
    gtf = pd.read_csv(gtf_fname, header=0, index_col=0)
    gtf = gtf.reindex(geneName.astype(str)) # sort as geneName
    logging.info(gtf.head())

    test_idx = gtf.chrom == chrom
    train_val_idx = gtf.chrom != chrom
    logging.info(np.sum(test_idx))

    x_test, y_test, gene_test = X[test_idx], y[test_idx], geneName[test_idx]
    x_train_val, y_train_val, gene_train_val = X[train_val_idx], y[train_val_idx], geneName[train_val_idx]

    x_train_idx, x_val_idx, y_train, y_val = train_test_split(range(len(x_train_val)), y_train_val, test_size=2000, shuffle=True, random_state=random_seed)
    x_train = x_train_val[x_train_idx]; x_val = x_train_val[x_val_idx]
    gene_train = gene_train_val[x_train_idx]; gene_val = gene_train_val[x_val_idx]

    logging.info(x_train.shape)
    logging.info(x_test.shape)
    logging.info(x_val.shape)

    h5file = h5py.File(output_dir, 'w')
    h5file.create_dataset("celltype", data=celltypes, dtype=h5py.special_dtype(vlen=str))
    h5file.create_dataset("annotation", data=anno, dtype=h5py.special_dtype(vlen=str), **compress_args)

    h5file.create_dataset("train_gene", data=gene_train, dtype=h5py.special_dtype(vlen=str), **compress_args)
    h5file.create_dataset("val_gene", data=gene_val, dtype=h5py.special_dtype(vlen=str), **compress_args)
    h5file.create_dataset("test_gene", data=gene_test, dtype=h5py.special_dtype(vlen=str), **compress_args)

    h5file.create_dataset("train_data", data=x_train, dtype=bool, **compress_args)
    h5file.create_dataset("train_label", data=y_train, dtype=data_type, **compress_args)

    h5file.create_dataset("val_data", data=x_val, dtype=bool, **compress_args)
    h5file.create_dataset("val_label", data=y_val, dtype=data_type, **compress_args)

    h5file.create_dataset("test_data", data=x_test, dtype=bool, **compress_args)
    h5file.create_dataset("test_label", data=y_test, dtype=data_type, **compress_args)

    h5file.close()

if mode == "leave_chrom_CV":

    gtf_fname, chrom = argv[-2:]
    chrom = str(chrom)
    gtf = pd.read_csv(gtf_fname, header=0, index_col=0)
    gtf = gtf.reindex(geneName.astype(str)) # sort as geneName
    logging.info(gtf.head())

    test_idx = gtf.chrom == chrom
    train_val_idx = gtf.chrom != chrom
    logging.info(np.sum(test_idx))

    x_test, y_test, gene_test = X[test_idx], y[test_idx], geneName[test_idx]
    x_train_val, y_train_val, gene_train_val = X[train_val_idx], y[train_val_idx], geneName[train_val_idx]

    kf = KFold(n_splits=10, random_state=random_seed, shuffle=True)
    fold = 0 
    for train_idx, val_idx in kf.split(range(len(x_train_val))):
        fold += 1
        logging.info("processing fold %d" % fold)

        x_train, x_val = x_train_val[train_idx], x_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        gene_train, gene_val = gene_train_val[train_idx], gene_train_val[val_idx]

        logging.info(x_train.shape)
        logging.info(x_test.shape)
        logging.info(x_val.shape)

        output_fname = os.path.join(output_dir, "Dataset.leave_chrom"+chrom+'_'+str(fold)+".h5")
        h5file = h5py.File(output_fname, 'w')
        h5file.create_dataset("celltype", data=celltypes, dtype=h5py.special_dtype(vlen=str))
        h5file.create_dataset("annotation", data=anno, dtype=h5py.special_dtype(vlen=str), **compress_args)

        h5file.create_dataset("train_gene", data=gene_train, dtype=h5py.special_dtype(vlen=str), **compress_args)
        h5file.create_dataset("val_gene", data=gene_val, dtype=h5py.special_dtype(vlen=str), **compress_args)
        h5file.create_dataset("test_gene", data=gene_test, dtype=h5py.special_dtype(vlen=str), **compress_args)

        h5file.create_dataset("train_data", data=x_train, dtype=bool, **compress_args)
        h5file.create_dataset("train_label", data=y_train, dtype=data_type, **compress_args)
        
        h5file.create_dataset("val_data", data=x_val, dtype=bool, **compress_args)
        h5file.create_dataset("val_label", data=y_val, dtype=data_type, **compress_args)
        
        h5file.create_dataset("test_data", data=x_test, dtype=bool, **compress_args)
        h5file.create_dataset("test_label", data=y_test, dtype=data_type, **compress_args)
        
        h5file.close()

# h5file = h5py.File(output_fname, 'r')
# h5file["test_data"][0]