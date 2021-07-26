import h5py, shutil
from sys import argv
import numpy as np

data_type = bool

def random_label(matrix):
    ratio = np.sum(matrix, axis=0).astype(int)
    new_arr = np.zeros_like(matrix)
    for idx in range(new_arr.shape[1]):
        new_arr[:ratio[idx], idx] = 1
        np.random.shuffle(new_arr[:, idx])
    return new_arr

dataset_fname, output_fname = argv[1:]
# "../nvwa-pse-official/data-hcl/HCL_pseudocell_bin.h5", \
# "train.human_random_label.h5"

# cp dataset_fname output_fname
shutil.copy(dataset_fname, output_fname)

# load nvwa-datasets
h5file = h5py.File(output_fname, 'r+')
y_train_nvwa = h5file["train_label"][:].astype(data_type)
y_val_nvwa = h5file["val_label"][:].astype(data_type)
y_test_nvwa = h5file["test_label"][:].astype(data_type)

y_train_random = random_label(y_train_nvwa)
y_val_random = random_label(y_val_nvwa)
y_test_random = random_label(y_test_nvwa)

# check
print(y_train_random.shape, y_test_random.shape, y_val_random.shape)
print("checked, saving...")

# save datasets
h5file["train_label"][:] = y_train_random
h5file["val_label"][:] = y_val_random
h5file["test_label"][:] = y_test_random
h5file.close()