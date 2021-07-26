import os, time, shutil, logging, argparse, itertools, sys
import pickle, h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize

from hyperopt import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("region")
parser.add_argument("--gpu-device", dest="device_id", default="0")
parser.add_argument("--trails", dest="trails", default="explainable")
parser.add_argument("--tower_by", dest="tower_by", default="Celltype")
parser.add_argument("--reverse_comp", dest="reverse_comp", action="store_true", default=False)

args = parser.parse_args()
data, region = args.data, args.region
device_id = args.device_id
trails_fname, tower_by = args.trails, args.tower_by # default
reverse_comp = args.reverse_comp

# set random_seed
set_random_seed()
set_torch_benchmark()

use_cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = device_id
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

# unpack datasets
h5file = h5py.File(data, 'r')
celltype = h5file["celltype"][:]
anno = h5file["annotation"][:]
anno = pd.DataFrame(anno, columns=["Cell", "Species", "Celltype", "Cluster"])
anno_cnt = anno.groupby(tower_by)["Species"].count()

y_test_onehot = h5file["test_label"][:2].astype(np.float32)
h5file.close()

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

# define hyperparams
output_size = y_test_onehot.shape[-1]
params["output_size"] = output_size

params['is_spatial_transform'] = True
params['anno_cnt'] = anno_cnt
params['globalpoolsize'] = 16

logging.info(params)

model = get_model(params)
model.load_state_dict(torch.load("./Log/best_model.pth", map_location=device))

model.to(device)
model.eval()

def one_hot(seq):
    seq_len = len(seq.item(0))
    seqindex = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
    seq_vec = np.zeros((len(seq),seq_len,4), dtype='bool')
    for i in range(len(seq)):
        thisseq = seq.item(i)
        for j in range(seq_len):
            try:
                seq_vec[i,j,seqindex[thisseq[j]]] = 1
            except:
                pass
    return seq_vec

table = pd.read_table(region, index_col=0, header=None)
seqs = one_hot(table.values)
if reverse_comp:
    seqs = np.array([seq[::-1,::-1] for seq in seqs])
logging.info(seqs.shape)

batch_size = 32
test_loader = DataLoader(list(zip(seqs.swapaxes(-1,1).astype(np.float32), itertools.cycle([0]))), batch_size=batch_size, 
                            shuffle=False, num_workers=0, drop_last=False)

preds = []
for data, _ in test_loader:
    data = data.to(device)
    pred = model(data).cpu().data.numpy()
    preds.append(pred)

preds = np.vstack(preds) 

# random_promoter = np.zeros((256, 4, 2000), dtype=np.float32)
# for idx in range(random_promoter.shape[0]):
#     label = np.random.randint(0, 4, random_promoter.shape[-1])
#     random_promoter[idx] = label_binarize(label, classes=range(4)).swapaxes(0,1)

# random_promoter = torch.from_numpy(random_promoter).to(device)
# background = model(random_promoter).cpu().data.numpy().mean(0)
# print(background.shape)

# N_promoter = np.zeros((256, 4, 10500), dtype=np.float32)
# N_promoter = torch.from_numpy(N_promoter).to(device)
# N_background = model(N_promoter).cpu().data.numpy().mean(0)
# print(N_background.shape)

predictions_test = preds
# predictions_test = preds - background
# predictions_test = preds - N_background
print(predictions_test.shape)

for i in range(predictions_test.shape[-1]):
    p_test = predictions_test[:, i].flatten()
    df = pd.DataFrame(np.column_stack((table.index, p_test)), columns=['Info','Pred'])
    if not reverse_comp:
        df.to_csv("./test_predict_"+str(i)+".Plus.txt", index=False, header=True, sep='\t')
    else:
        df.to_csv("./test_predict_"+str(i)+".Minus.txt", index=False, header=True, sep='\t')