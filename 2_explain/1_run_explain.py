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
parser.add_argument("--globalpoolsize", dest="globalpoolsize", default=None, type=int)
parser.add_argument("--tower_hidden", dest="tower_hidden", default=None, type=int)
parser.add_argument("--pos", dest="pos", default=None, type=int)

args = parser.parse_args()
data = args.data
device_id, mode, trails_fname, use_data_rc_augment = args.device_id, args.mode, args.trails, args.use_data_rc_augment # default
patience = args.patience
tower_by = args.tower_by
pos = args.pos
globalpoolsize = args.globalpoolsize
tower_hidden = args.tower_hidden

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

# x_train = h5file["train_data"][:].astype(np.float32)
# y_train_onehot = h5file["train_label"][:].astype(np.float32)
# x_val = h5file["val_data"][:].astype(np.float32)
# y_val_onehot = h5file["val_label"][:].astype(np.float32)

x_test = h5file["test_data"][:].astype(np.float32)
y_test_onehot = h5file["test_label"][:].astype(np.float32)

# x_test = np.vstack([x_train, x_val, x_test])
# y_test_onehot = np.vstack([y_train_onehot, y_val_onehot, y_test_onehot])

# train_gene = h5file["train_gene"][:]
# val_gene = h5file["val_gene"][:]
test_gene = h5file["test_gene"][:]
# test_gene = np.hstack([train_gene, val_gene, test_gene])
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

if pos:
    params['leftpos'] = 10000 - pos
    params['rightpos'] = 10000 + pos

logging.info(params)

# define datasets parameters
leftpos = int(params['leftpos'])
rightpos = int(params['rightpos'])
logging.info((leftpos, rightpos))

# define hyperparams
output_size = y_test_onehot.shape[-1]
params["output_size"] = output_size
if globalpoolsize:
    params['globalpoolsize'] = globalpoolsize
if tower_hidden:
    params['tower_hidden'] = tower_hidden

# define dataset params
batch_size = int(params['batchsize'])

# test_loader
x_test = x_test[:, :, leftpos:rightpos]
logging.info(x_test.shape)

test_loader = DataLoader(list(zip(x_test, y_test_onehot)), batch_size=500, 
                            shuffle=False, num_workers=0, drop_last=False)

model = get_model(params)
model.load_state_dict(torch.load("./Log/best_model.pth", map_location=device), strict=True)
# model.Embedding.load_state_dict(torch.load("./Log/best_model.pth", map_location=device), strict=True)
# model.MLP.load_state_dict(torch.load("./Log/best_model.pth", map_location=device), strict=True)
# model = torch.load("./Log/best_model.p")
model.to(device)
model.eval()
logging.info(model.__str__())

# get motif
# convolution layer 1
fmap1, X1 = get_fmap(model, model.Embedding.conv1[0], test_loader)

# t = 0.99
# W1 = get_activate_W_from_fmap(fmap1, X1, pool=1, threshold=0.99, motif_width=7)
# meme_generate(W1, output_file="./Motif/meme_conv1_thres99.txt")

# W1_freq, W1_IC = calc_frequency_W(W1, background=0.25)
# pd.DataFrame({"freq":W1_freq, "IC":W1_IC}).to_csv("./Motif/meme_conv1_thres99_IC_freq.csv")

# t = 0.9
W1 = get_activate_W_from_fmap(fmap1, X1, pool=1, threshold=0.9, motif_width=7)
meme_generate(W1, output_file="./Motif/meme_conv1_thres9.txt")

W1_freq, W1_IC = calc_frequency_W(W1, background=0.25)
pd.DataFrame({"freq":W1_freq, "IC":W1_IC}).to_csv("./Motif/meme_conv1_thres9_IC_freq.csv")

# t = 0.8
# W1 = get_activate_W_from_fmap(fmap1, X1, pool=1, threshold=0.8, motif_width=7)
# meme_generate(W1, output_file="./Motif/meme_conv1_thres8.txt")

# W1_freq, W1_IC = calc_frequency_W(W1, background=0.25)
# pd.DataFrame({"freq":W1_freq, "IC":W1_IC}).to_csv("./Motif/meme_conv1_thres8_IC_freq.csv")

# # t = 0.5
# W1 = get_activate_W_from_fmap(fmap1, X1, pool=1, threshold=0.5, motif_width=7)
# meme_generate(W1, output_file="./Motif/meme_conv1_thres5.txt")

# # motif frequency
# W1_freq, W1_IC = calc_frequency_W(W1, background=0.25)
# pd.DataFrame({"freq":W1_freq, "IC":W1_IC}).to_csv("./Motif/meme_conv1_thres5_IC_freq.csv")

# gene_seq, gene_name = get_activate_sequence_from_fmap(fmap1, X1, pool=1, threshold=0.8, motif_width=20)
# onehot2seq(gene_seq, gene_name, "./Motif/test_gene_activate_conv1.fasta")

logging.info("conv layer 1 finished")

# convolution layer 2
# W2 = get_activate_W(model, model.Embedding.conv2[0], test_loader, pool=7, threshold=0.8, motif_width=7)
# meme_generate(W2, output_file="./Motif/meme_conv2.txt")

# out_fname = "./Motif/test_gene_activate_conv2.fasta"
# save_activate_sequence(model, model.Embedding.conv2[0], test_loader, out_fname, pool=7, motif_width=10)

# logging.info("conv layer 2 finished")

# get feature map of FCLayer
# fmap_dense1, _ = get_fmap(model, model.MLP.fc1[0], test_loader)
# pd.DataFrame(fmap_dense1, index=test_gene).to_csv("./Motif/fmap_dense1.csv")
# logging.info("Dense layer finished")

# fmap_task_cat, _ = get_fmap(model, model.cat_task, test_loader)
# pd.DataFrame(fmap_task_cat, index=test_gene, columns=celltype).to_pickle("./Motif/fmap_task_cat.p", compression='xz')
# logging.info("task_cat layer finished")

# channel influence
channel_influence = channel_target_influence(model, model.Embedding.conv1[0], test_loader, device)
df = pd.DataFrame(channel_influence, columns=celltype)
df.to_pickle("./Motif/influence_conv1.p", compression='xz')
df.mean(1).to_csv("./Motif/influence_conv1_mean.csv")
logging.info("conv1 channel influence finished")

# channel_influence = channel_target_influence(model, model.Embedding.conv2[0], test_loader, device)
# df = pd.DataFrame(channel_influence, columns=celltype)
# df.to_pickle("./Motif/influence_conv2.p", compression='xz')
# df.mean(1).to_csv("./Motif/influence_conv2_mean.csv")
# logging.info("conv2 channel influence finished")

# channel_influence = channel_target_influence(model, model.MLP.fc1[0], test_loader, device)
# df = pd.DataFrame(channel_influence, columns=celltype)
# df.to_pickle("./Motif/influence_MLP.p", compression='xz')
# df.mean(1).to_csv("./Motif/influence_MLP_mean.csv")
# logging.info("MLP channel influence finished")

# channel_influence = channel_target_influence(model, model.task_Drosophila10.fc1[0], test_loader, device)
# df = pd.DataFrame(channel_influence, columns=celltype)
# df.to_pickle("./Motif/influence_task_Drosophila10.p", compression='xz')
# df.mean(1).to_csv("./Motif/influence_task_Drosophila10_mean.csv")
# logging.info("task_Drosophila10 channel influence finished")

out_channels = model.Embedding.conv1[0].out_channels
channel_influence = layer_channel_combination_influence(model, model.Embedding.conv1[0], test_loader, device)
df = pd.DataFrame(channel_influence, columns=celltype, index=itertools.combinations(range(out_channels), 2))
df.to_pickle("./Motif/influence_layer1_combination.p", compression='xz')
df.mean(1).to_csv("./Motif/influence_layer1_combination_mean.csv")
logging.info("influence_layer1_combination finished")

# convert to cpu 
# device = torch.device("cpu")
# model.to(device)

# input_tensor = torch.from_numpy(x_test[np.random.randint(0, x_test.shape[0], 256),:,:]).to(device)
# # saliancy score
# saliency_length = input_saliancy_location(model, input_tensor, n_class=len(celltype), use_abs=True)
# saliency_length.to_csv("./Motif/saliency_location.csv")

# layer conductance
# df_conv1 = label_neuron_importance(model, model.Embedding.conv1[0], input_tensor, label=celltype)
# df_conv1.to_csv("./importance_conv1.csv")

# df_conv2 = label_neuron_importance(model, model.Embedding.conv2[0], input_tensor, label=celltype)
# df_conv2.to_csv("./Motif/importance_conv2.csv")