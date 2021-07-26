import h5py, shutil
from sys import argv
import numpy as np
from sklearn.preprocessing import label_binarize

data_type = bool

def random_features(matrix):
    new_arr = np.zeros_like(matrix)
    for idx in range(matrix.shape[0]):
        label = np.random.randint(0, 4, new_arr.shape[-1])
        new_arr[idx] = label_binarize(label, classes=range(4)).swapaxes(0,1)
    return new_arr


dataset_fname, output_fname = argv[1:]
# "../nvwa-pse-official/data-hcl/HCL_pseudocell_bin.h5", \
# "train.human_random_label.h5"

# cp dataset_fname output_fname
shutil.copy(dataset_fname, output_fname)

# load nvwa-datasets
h5file = h5py.File(output_fname, 'r+')
x_train_nvwa = h5file["train_data"][:].astype(data_type)
x_val_nvwa = h5file["val_data"][:].astype(data_type)
x_test_nvwa = h5file["test_data"][:].astype(data_type)

x_train_random = random_features(x_train_nvwa)
x_val_random = random_features(x_val_nvwa)
x_test_random = random_features(x_test_nvwa)

# check
print(x_train_random.shape, x_val_random.shape, x_test_random.shape)
print("checked, saving...")

# save datasets
h5file["train_data"][:] = x_train_random
h5file["val_data"][:] = x_val_random
h5file["test_data"][:] = x_test_random
h5file.close()