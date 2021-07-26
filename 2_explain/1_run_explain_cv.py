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
species = os.path.basename(data).split('.')[1]
logging.info("#"*60)
logging.info("switching datasets: %s" % species)

# unpack datasets
h5file = h5py.File(data, 'r')
celltype = h5file["celltype"][:]
anno = h5file["annotation"][:]
anno = pd.DataFrame(anno, columns=["Cell", "Species", "Celltype", "Cluster"])
anno_cnt = anno.groupby(tower_by)["Species"].count()

x_test = h5file["test_data"][:].astype(np.float32)
y_test_onehot = h5file["test_label"][:].astype(np.float32)
test_gene = h5file["test_gene"][:]
h5file.close()

logging.info(x_test.shape)
logging.info(y_test_onehot.shape)

# trails
if trails_fname == "best_manual":
    params = best_params
    logging.info("using best manual model")
elif trails_fname == "explainable":
    params = explainable_params
    logging.info("using explainable model")
elif trails_fname == "NIN":
    params = explainable_params
    params["is_NIN"] = True
    logging.info("using NIN model")
else:
    trials = pickle.load(open(trails_fname, 'rb'))
    best = trials.argmin
    params = space_eval(param_space, best)
    logging.info("using model from trails:\t%s", trails_fname)

params['is_spatial_transform'] = False
params['anno_cnt'] = anno_cnt
logging.info(params)

# define datasets parameters
leftpos = int(params['leftpos'])
rightpos = int(params['rightpos'])
logging.info((leftpos, rightpos))

# define hyperparams
output_size = y_test_onehot.shape[-1]
params["output_size"] = output_size

# define dataset params
batch_size = int(params['batchsize'])

# test_loader
x_test = x_test[:, :, leftpos:rightpos]
logging.info(x_test.shape)

test_loader = DataLoader(list(zip(x_test, y_test_onehot)), batch_size=32, 
                            shuffle=False, num_workers=0, drop_last=False)

model = get_model(params)
model.load_state_dict(torch.load("./Log/best_model.pth", map_location=device), strict=False)
model.to(device)
model.eval()
logging.info(model.__str__())

# get motif
## convolution layer 1
fmap1, X1 = get_fmap(model, model.Embedding.conv1[0], test_loader)
W1 = get_activate_W_from_fmap(fmap1, X1, pool=1, threshold=0.9, motif_width=7)
meme_generate(W1, output_file="./Motif/meme_conv1_thres9.txt")

gene_seq, gene_name = get_activate_sequence_from_fmap(fmap1, X1, pool=1, threshold=0.99, motif_width=9)
onehot2seq(gene_seq, gene_name, "./Motif/test_gene_activate_conv1_t95.fasta")

# # motif frequency
# W1_freq, W1_IC = calc_frequency_W(W1, background=0.25)
# pd.DataFrame({"freq":W1_freq, "IC":W1_IC}).to_csv("./Motif/W1_IC_freq.csv")
# logging.info("conv layer 1 finished")

# # convolution layer 2
# W2 = get_activate_W(model, model.Embedding.conv2[0], test_loader, pool=7, threshold=0.8, motif_width=7)
# meme_generate(W2, output_file="./Motif/meme_conv2.txt")
# logging.info("conv layer 2 finished")