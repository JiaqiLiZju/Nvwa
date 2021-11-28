import os, time, shutil, logging, argparse
import pickle, h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from sklearn.metrics import *
from hyperopt import *

from utils import *
from explainer import *
os.makedirs("Motif", exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("--mode", dest="mode", default="explain")
parser.add_argument("--gpu-device", dest="device_id", default="0")
parser.add_argument("--trails", dest="trails", default="explainable")
parser.add_argument("--tower_by", dest="tower_by", default="Celltype")
parser.add_argument("--use_data_rc_augment", dest="use_data_rc_augment", action="store_true", default=False)
parser.add_argument("--patience", dest="patience", default=10, type=int)

args = parser.parse_args()
data = args.data
device_id, mode, trails_fname, use_data_rc_augment = args.device_id, args.mode, args.trails, args.use_data_rc_augment # default
patience = args.patience
tower_by = args.tower_by

# set random_seed
set_random_seed()
set_torch_benchmark()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime('./log_HyperBest.explain.%m%d.%H:%M:%S.txt'),
                    filemode='w')

## change
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
logging.info(device)

# unpack datasets
h5file = h5py.File(data, 'r')
gene_features = h5file["gene_features"][:]
# gene_target = h5file["gene_target"][:]

x_test = h5file["test_data"][:].astype(np.float32)
y_test = onehot_encode(h5file["test_label"][:].astype(int)).astype(np.float32)
cells_test = h5file["cells_test"][:]

logging.info(x_test.shape)
logging.info(y_test.shape)

h5file.close()

features_size = x_test.shape[-1]
target_size = y_test.shape[-1]

# test_loader
test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=500, 
                            shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

model = MLP(dense_input_size=features_size, 
            dense1=512, dropout1=0.5, 
            is_fc2=True, dense2=128, dropout2=0.3,
            is_pred=True, output_size=target_size, pred_prob=False)

model.load_state_dict(torch.load("./Log/best_model.pth", map_location=device), strict=True)
logging.info("weights inited and embedding weights loaded")

model.eval()
model.to(device)
logging.info(model.__str__())

# channel influence
channel_influence = input_channel_target_influence(model, model.fc1[0], test_loader, device)
df = pd.DataFrame(channel_influence, index=gene_features)
df.to_pickle("./Motif/influence_tfs.p", compression='xz')
df.mean(1).to_csv("./Motif/influence_tfs_mean.csv")
logging.info("tfs influence finished")

# channel_influence = input_layer_channel_combination_influence(model, model.fc1[0], test_loader, device)
# df = pd.DataFrame(channel_influence, columns=cells_test, index=itertools.combinations(gene_features, 2))
# df.to_pickle("./Motif/influence_tfs_combination.p", compression='xz')
# df.mean(1).to_csv("./Motif/influence_tfs_combination_mean.csv")
# logging.info("tfs_combination finished")
