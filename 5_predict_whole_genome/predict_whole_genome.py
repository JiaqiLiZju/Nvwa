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

def group_celltype(expr, anno):
    expr = pd.DataFrame(expr.T)
    expr.index = anno.Cell
    
    expr = expr.groupby(anno.Celltype.values).mean().T.values
    return expr

def plot_corr_heatmap(expr, anno, output_fname="out.pdf"):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    color = ("#E6AB02",  "#66A61E", "#D95F02", "#1B9E77", "#E7298A",  "#E31A1C", "#A6761D"  , "#B2DF8A",   "#FFFF99",   "#7570B3", "#FF7F00",  "#A65628", "#B3CDE3", "#BC80BD",     "#A6CEE3","#984EA3",   "#CCEBC5",  "#E41A1C",    "#4DAF4A","#BEBADA", "#B3DE69", "#CAB2D6","#FFFFB3",   "#33A02C","#B15928", "#6A3D9A","#FBB4AE",    "blue",          "#FB8072",      "#FFFF33","#CCEBC5",      "#A6761D",   "#2c7fb8","#fa9fb5",  "#BEBADA","#E7298A", "#E7298A" )
    regions = ("Secretory", "Muscle", "Neuron" , "Immune", "Epithelial", "Glia", "Proliferating","Other",  "Germline","Stromal","Phagocytes","MAG","Rectum", "Coelomocytes","Intestine","Hepatocyte","Germ","Endothelial","Erythroid","Testis","Mesenchyme","Yolk", "Midgut" ,"Embryo","Hemocytes",  "Fat",  "Unknown","Gastrodermis","DigFilaments","Pigment","BasementMembrane","Endoderm","RP_high","FatBody","Male","Nephron", "Pancreatic")
    color_regions = {x:y for x,y in zip(regions, color)}
    color_regions

    anno_color = anno
    anno_color["colors_lineage"] = anno_color[['Cluster']].applymap(lambda x: color_regions[x])
    anno_color

    celltype_anno = anno_color[["Celltype", "Cluster", "colors_lineage"]].drop_duplicates().set_index(["Celltype"]).loc[np.unique(anno.Celltype.values)].colors_lineage
    celltype_anno

    lut = {cluster:color_regions.get(cluster) for cluster in anno_color.Cluster.unique()}
    lut

    corr = np.corrcoef(expr, expr, rowvar=False)
    corr_pt = pd.DataFrame(corr[:expr.shape[1],expr.shape[1]:], index=np.unique(anno.Celltype.values), columns=np.unique(anno.Celltype.values))

    plt.figure(figsize=(15,15))
    g = sns.clustermap(corr_pt, cmap='vlag', 
                       xticklabels=False, yticklabels=False, cbar_pos=None,
                       col_colors=celltype_anno, row_colors=celltype_anno,
                       dendrogram_ratio=(.1, .1), colors_ratio=0.02,
    #                standard_scale=1,
    #                z_score=1
                  )

    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut, title='CellLieange',
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

    plt.savefig(output_fname)
    plt.show()
    plt.close()

# read sequence
table = pd.read_table(region, index_col=0, header=None)
blank = 'N'*13000
blank_idx = (table[[1]] == blank).values.flatten()

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
preds[blank_idx,:] = 0

predictions_test = group_celltype(preds, anno)
# bs = 10000
# predictions_test = []
# for idx in range(0, preds.shape[0], bs):
#     preds_batch = preds[idx:idx+bs,:]
#     predictions_test.append(group_celltype(preds_batch, anno))
# predictions_test = np.vstack(predictions_test)

plot_corr_heatmap(pd.DataFrame(predictions_test, columns=np.unique(anno.Celltype)), anno, output_fname="./predictions_corr_heatmap.pdf")

for i in range(predictions_test.shape[-1]):
    p_test = predictions_test[:, i].flatten()
    df = pd.DataFrame(np.column_stack((table.index, p_test)), columns=['Info','Pred'])
    if not reverse_comp:
        df.to_csv("./test_predict_"+str(i)+".Plus.txt", index=False, header=True, sep='\t')
    else:
        df.to_csv("./test_predict_"+str(i)+".Minus.txt", index=False, header=True, sep='\t')

# random promoter
random_promoter = np.zeros((10000, 4, 13000), dtype=np.float32)
for idx in range(random_promoter.shape[0]):
    label = np.random.randint(0, 4, random_promoter.shape[-1])
    random_promoter[idx] = label_binarize(label, classes=range(4)).swapaxes(0,1)

print(random_promoter.shape)

test_loader = DataLoader(list(zip(random_promoter, itertools.cycle([0]))), batch_size=batch_size, 
                            shuffle=False, num_workers=0, drop_last=False)

preds = []
for data, _ in test_loader:
    data = data.to(device)
    pred = model(data).cpu().data.numpy()
    preds.append(pred)

background = np.vstack(preds) 
background = group_celltype(background, anno)
background = pd.DataFrame(background, columns=np.unique(anno.Celltype))
background.to_csv("./code_test/prdicted_null/null_random_background.csv")
plot_corr_heatmap(background, anno, output_fname="./code_test/prdicted_null/null_random_background.pdf")

# random location
chrom, start, end = 'chr8', 0, 129401213
random_loc = random.sample(range(start, end), 10000)

import pysam
genome = "/media/ggj/Files/NvWA/PreprocGenome/database_genome/Mouse_GRCm/Mus_musculus.GRCm38.88.fasta"
fa_handler = pysam.FastaFile(genome)

loc_str, seqs = [],[]
for loc in random_loc:
    fp_start = loc - 6500 - 1 # -1 for pysam coord
    bp_end = loc + 6500 - 1
    if fp_start < start or bp_end > end:
        continue
        
    seq = fa_handler.fetch(chrom, start=fp_start, end=bp_end)
    seqs.append(seq)
    loc_str.append("%s:%d-%d"%(chrom, fp_start, bp_end))
    
fa_handler.close()

table = pd.DataFrame(seqs, index=loc_str)
seqs = one_hot(table.values)

test_loader = DataLoader(list(zip(seqs.swapaxes(-1,1).astype(np.float32), itertools.cycle([0]))), batch_size=batch_size, 
                            shuffle=False, num_workers=0, drop_last=False)

preds = []
for data, _ in test_loader:
    data = data.to(device)
    pred = model(data).cpu().data.numpy()
    preds.append(pred)

preds = np.vstack(preds) 
preds = group_celltype(preds, anno)
preds = pd.DataFrame(preds, index=table.index, columns=np.unique(anno.Celltype))
preds.to_csv("./preds_random_location.csv")
plot_corr_heatmap(preds, anno, output_fname="./preds_random_location.pdf")


# N-blank promoter
# N_promoter = np.zeros((10000, 4, 13000), dtype=np.float32)
# batch_size = 128
# test_loader = DataLoader(list(zip(N_promoter, itertools.cycle([0]))), batch_size=batch_size, 
#                             shuffle=False, num_workers=0, drop_last=False)

# preds = []
# for data, _ in test_loader:
#     data = data.to(device)
#     pred = model(data).cpu().data.numpy()
#     preds.append(pred)

# N_background = np.vstack(preds) 
# N_background = group_celltype(N_background, anno)
# N_background = pd.DataFrame(N_background, columns=np.unique(anno.Celltype))
# N_background.to_csv("./code_test/prdicted_null/null_blank_N_background.csv")
