import h5py, shutil
import numpy as np
from sys import argv

dataset_fname, output_fname = argv[1:]

fhr = h5py.File(dataset_fname, 'r')
h5file = h5py.File(output_fname, 'w')

# save datasets
h5file.create_dataset("test_headers", data=fhr["celltype"][:])

h5file.create_dataset("train_in", data=fhr["train_data"][:][:,:,None,:].astype(bool))
h5file.create_dataset("valid_in", data=fhr["val_data"][:][:,:,None,:].astype(bool))
h5file.create_dataset("test_in", data=fhr["test_data"][:][:,:,None,:].astype(bool))

h5file.create_dataset("train_out", data=fhr["train_label"][:].astype(bool))
h5file.create_dataset("valid_out", data=fhr["val_label"][:].astype(bool))
h5file.create_dataset("test_out", data=fhr["test_label"][:].astype(bool))

h5file.close()
fhr.close()

# # cp dataset_fname output_fname
# shutil.copy(dataset_fname, output_fname)

# # load nvwa-datasets
# h5file = h5py.File(output_fname, 'r+')

# # save datasets
# h5file["test_headers"] = h5file["celltype"]

# h5file["train_in"] = h5file["train_data"][:][:,:,None,:]
# h5file["valid_in"] = h5file["val_data"][:][:,:,None,:]
# h5file["test_in"] = h5file["test_data"][:][:,:,None,:]

# h5file["train_out"] = h5file["train_label"]
# h5file["valid_out"] = h5file["val_label"]
# h5file["test_out"] = h5file["test_label"]

# del h5file["train_data"], h5file["val_data"], h5file["test_data"]
# del h5file["train_label"], h5file["val_label"], h5file["test_label"]

# h5file.close()