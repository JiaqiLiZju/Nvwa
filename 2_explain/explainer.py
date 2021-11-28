import os, logging, itertools
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from copy import deepcopy
from tqdm import tqdm
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from captum.attr import *

# motif analysis
def calc_motif_IC(motif, background=0.25):
    """IC Bernouli"""
    H = (motif * np.log2(motif / background + 1e-6)).sum()
    logging.info("Motif IC(Bernouli): %.4f" % H)
    return H

def info_content(pwm, bg=0.5):
    pseudoc = 1e-6
    bg_pwm = [1-bg, bg, bg, 1-bg]
    
    ic = 0 
    for i in range(pwm.shape[0]):
        for j in range(4):
            ic += -bg_pwm[j]*np.log2(bg_pwm[j]) + pwm[i][j]*np.log2(pseudoc + pwm[i][j])
    return ic

def calc_motif_entropy(motif, background=0.25):
    '''Entropy'''
    H = -(motif * np.log2(motif / background + 1e-6)).sum()
    logging.info("Motif Entropy: %.4f" % H)
    return H

def calc_motif_frequency(motif_IC):
    f = np.power(2, -(motif_IC - 1))
    logging.info("Motif Frequency: %.4f" % f)
    return f

def calc_frequency_W(W, background=0.25):
    motif_frequency_l, motif_IC_l = [], []
    for pwm in W:
        pwm = normalize_pwm(pwm)
        motif_IC = calc_motif_IC(pwm)
        motif_freq = calc_motif_frequency(motif_IC)
        motif_IC_l.append(motif_IC); motif_frequency_l.append(motif_freq)
    return motif_frequency_l, motif_IC_l

# get motif directly from convolution parameters
def get_W_from_conv(model, hook_module):
    # weights = list(model.hook_module.parameters())[0]
    # weights = weights.cpu().data.numpy()
    # W = weights[:,:,::-1]
    # # normalize counts
    # seq_align = (np.sum(seq_align, axis=0)/np.sum(count_matrix, axis=0))*np.ones((4,motif_width*pool))
    # seq_align[np.isnan(seq_align)] = 0
    # W.append(seq_align)
    raise NotImplementedError

# hook
class ActivateFeaturesHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()#.mean(-1)
    def get_features(self):
        return self.features
    def close(self):
        self.hook.remove()

def get_fmap(model, hook_module, data_loader, device=torch.device("cuda")):
    fmap, X = [], []
    model.eval()
    with torch.no_grad():
        activations = ActivateFeaturesHook(hook_module)
        for x_tensor, _ in data_loader:
            x_tensor = x_tensor.to(device)
            _ = model(x_tensor)
            X.append(x_tensor.cpu().numpy())
            fmap.append(activations.get_features())
        fmap = np.vstack(fmap)
        X = np.vstack(X)
        activations.close()
    return fmap, X

def onehot2seq(gene_seq, gene_name, out_fname):
    d = {0:'A', 1:'C', 2:'G', 3:'T'}
    s = ''
    for i, fas in zip(gene_name, map(lambda y: ''.join(map(lambda x:d[x], np.where(y.T==1)[-1])), gene_seq)):
        s += '>'+str(i)+'\n'
        s += fas+'\n'
    with open(out_fname, 'w') as fh:
        fh.write(s)

def get_activate_sequence_from_fmap(fmap, X, pool=1, threshold=0.99, motif_width=40):
    motif_nb = fmap.shape[1]
    seq_len = X.shape[-1]

    W, M = [], []
    for filter_index in range(motif_nb):
        # find regions above threshold
        data_index, pos_index = np.where(fmap[:,filter_index,:] > np.max(fmap[:,filter_index,:], axis=1, keepdims=True)*threshold)

        for i in range(len(pos_index)):
            # handle boundary conditions
            start = pos_index[i] - 1
            end = pos_index[i] + motif_width + 2
            if end > seq_len:
                end = seq_len
                start= end - motif_width - 2 
            if start < 0:
                start = 0 
                end = start + motif_width + 2

            seq = X[data_index[i], :, start*pool:end*pool]
            W.append(seq)
            M.append('_'.join(("Motif", str(filter_index), "Act", str(i))))

    return W, M

def save_activate_sequence(model, hook_module, data, out_fname, pool=1, threshold=0.99, motif_width=40):
    fmap, X = get_fmap(model, hook_module, data)
    gene_seq, gene_name = get_activate_sequence_from_fmap(fmap, X, pool=pool, threshold=threshold, motif_width=motif_width)
    onehot2seq(gene_seq, gene_name, out_fname)

def get_activate_index_from_fmap(fmap, X, threshold=0.99):
    motif_nb = fmap.shape[1]
    X_dim, seq_len = X.shape[1], X.shape[-1]

    W={}
    for filter_index in range(motif_nb):
        # find regions above threshold
        data_index, pos_index = np.where(fmap[:,filter_index,:] > np.max(fmap[:,filter_index,:], axis=1, keepdims=True)*threshold)
        W[filter_index] = [data_index, pos_index]

    return W

def get_activate_regulon_fromW(W, gene_name=None, threshold=0.99, device=torch.device("cuda")):
    # fmap, X = get_fmap(model, hook_module, data, device=device)
    # W = get_activate_index_from_fmap(fmap, X, threshold=threshold)
    
    regulon, regulon_pos = {}, {}
    for k,v in W.items():
        gene_idx, pos_idx = deepcopy(v[0]), deepcopy(v[1])
        mask = (4500<pos_idx) & (pos_idx<8500) # -2k < pos < 2k
        gene_idx = gene_idx[mask]
        pos_idx = pos_idx[mask]

        gene_id, cnt = np.unique(gene_idx, return_counts=True)
        t = np.mean(cnt) + 3 * np.std(cnt)
        # t = np.quantile(cnt, 0.99)
        # t = np.max(cnt) * 0.4
        idx = np.where(cnt >= t)
        target_gene_id = gene_id[idx]
        regulon[k] = gene_name[target_gene_id]

        target_mask = [g in target_gene_id for g in gene_idx]
        target_gene = gene_idx[target_mask]
        target_pos = pos_idx[target_mask]
        regulon_pos[k] = list(zip(gene_name[target_gene], target_pos))
    
    motif, target, pos = [], [], []
    for k,v in regulon_pos.items():
        for t,p in v:
            motif.append(k)
            target.append(t)
            pos.append(p)
    regulon_pos_df = pd.DataFrame({"Filter": motif, "target": target, "position":pos})

    return regulon, regulon_pos, regulon_pos_df

def prune_regulon_by_expr_influe_correlation(regulon, influe, expr_adata, threshold=0.1):
    motif, target, corr = [], [], []
    for f,v in tqdm(regulon.items()):
        influe_filter = influe.iloc[f,:].values
        
        for t in v:
            motif.append(f)
            target.append(t)
            
            expr_gene = expr_adata[:,t].X.flatten()
            corr.append(pearsonr(influe_filter, expr_gene)[0])

    regulon_df = pd.DataFrame({"Filter": motif, "target": target, "Corr": corr}).fillna(0)
    pruned_regulon_df = regulon_df[regulon_df.Corr > threshold]

    return regulon_df, pruned_regulon_df

def calculate_filter_pairs_jaccard(regulon_df):
    filter_nb = 128
    filter_pairs = list(itertools.combinations(range(filter_nb), 2))

    f1_l, f2_l, j_l = [], [], []
    for f1, f2 in tqdm(filter_pairs):
        t1 = set(regulon_df[regulon_df.Filter==f1].target.values)
        t2 = set(regulon_df[regulon_df.Filter==f2].target.values)

        inter = t1 & t2
        outer = t1 | t2

        jaccard_score = len(inter) / len(outer)
        
        f1_l.append(f1)
        f2_l.append(f2)
        j_l.append(jaccard_score)

    filter_pairs_jaccard = pd.DataFrame({"Filter1":f1_l, "Filter2":f2_l, "Jaccard":j_l})
    filter_pairs_jaccard["Pair"] = '(' + filter_pairs_jaccard.Filter1.astype("str") + "," + filter_pairs_jaccard.Filter2.astype("str") + ')'

    return filter_pairs_jaccard

def trim_duplicate_filter_pairs(filter_pairs_jaccard):
    tomtom = pd.read_csv("../../../Cross_Species_Motif/DD/tomtom.tsv", sep='\t', header=0, skipfooter=3, engine='python')
    tomtom = tomtom[["Query_ID", "Target_ID", "q-value"]]
    tomtom.columns = ["Filter1", "Filter2", "q_value"]
    tomtom['Pair'] = '(' + tomtom.Filter1.map(lambda x:x.split('_')[-1]) + ',' + tomtom.Filter2.map(lambda x:x.split('_')[-1]) + ')'
    mask = [pair not in tomtom.Pair.values for pair in filter_pairs_jaccard.Pair.values]
    # np.sum(mask)
    filter_pairs_jaccard_nonRedundant = filter_pairs_jaccard[mask]
    return filter_pairs_jaccard_nonRedundant

def trim_duplicate_filters():
    tomtom = pd.read_csv("../../Cross_Species_Motif/DD/tomtom.tsv", sep='\t', header=0, skipfooter=3, engine='python')
    tomtom = tomtom[["Query_ID", "Target_ID", "q-value"]]
    tomtom.columns = ["Filter1", "Filter2", "q_value"]
    tomtom['Pair'] = '(' + tomtom.Filter1.map(lambda x:x.split('_')[-1]) + ',' + tomtom.Filter2.map(lambda x:x.split('_')[-1]) + ')'
    tomtom = tomtom[tomtom["q_value"] < 0.01]

    IC = pd.read_csv("./Motif_filter/meme_conv1_thres9_IC_freq.csv", index_col=0)
    IC.index = IC.index.map(lambda x: "Motif_"+str(x))
    
    tomtom_ic = tomtom.merge(IC, left_on="Filter2", right_index=True, sort=False, how="left")
    nondup_filter = tomtom_ic[tomtom_ic["rank"]==1]
    return nondup_filter.Filter2.unique()

def regulon_combination(filter_pairs_jaccard_nonRedundant):
    t = np.mean(filter_pairs_jaccard_nonRedundant.Jaccard) + 3 * np.std(filter_pairs_jaccard_nonRedundant.Jaccard)
    filter_pairs_jaccard_nonRedundant = filter_pairs_jaccard_nonRedundant[filter_pairs_jaccard_nonRedundant.Jaccard > t]
    # filter_pairs_jaccard_nonRedundant.to_csv("./Motif/filter_pairs_jaccard_nonRedundant.pos2k.mean+3std.csv")
    return filter_pairs_jaccard_nonRedundant

def prune_regulonComb_by_expr_influeComb_correlation(filter_pairs, regulon_df, influe_comb, expr_adata, threshold=0.1):
    celltype, motif, target, corr = [], [], [], []
    for c in anno.Cellcluster.unique():
        c_idx = anno.Cellcluster == c
        
        for f1, f2, f in tqdm(filter_pairs[["Filter1", "Filter2", "Pair"]].values):

            t1 = regulon_df[regulon_df.Filter==f1].target.values
            t2 = regulon_df[regulon_df.Filter==f2].target.values
            t_inter = np.intersect1d(t1, t2)

            influe_comb_filter = influe_comb.loc[f, c_idx].values

            for t in t_inter:
                celltype.append(c)
                motif.append(f)
                target.append(t)

                expr_gene = expr[c_idx, t].X.flatten()
                corr.append(pearsonr(influe_comb_filter, expr_gene)[0])

    return regulon_df, pruned_regulon_df

def regulon_combination_pos(filter_pairs_jaccard_nonRedundant, regulon_pos_df):
    raise NotImplementedError

def get_activate_W_from_fmap(fmap, X, pool=1, threshold=0.99, motif_width=10):
    """
    get learned motif pwm based on motif_width
    """
    motif_nb = fmap.shape[1]
    X_dim, seq_len = X.shape[1], X.shape[-1]

    W=[]
    for filter_index in range(motif_nb):
        # find regions above threshold
        data_index, pos_index = np.where(fmap[:,filter_index,:] > np.max(fmap[:,filter_index,:], axis=1, keepdims=True)*threshold)

        seq_align = []; count_matrix = []
        for i in range(len(pos_index)):
            # pad 1-nt
            start = pos_index[i] - 1
            end = start + motif_width + 2
            # handle boundary conditions
            if end > seq_len:
                end = seq_len
                start = end - motif_width - 2 
            if start < 0:
                start = 0 
                end = start + motif_width + 2 

            seq = X[data_index[i], :, start*pool:end*pool]
            seq_align.append(seq)
            count_matrix.append(np.sum(seq, axis=0, keepdims=True))

        seq_align = np.array(seq_align)
        count_matrix = np.array(count_matrix)

        # normalize counts
        seq_align = (np.sum(seq_align, axis=0)/np.sum(count_matrix, axis=0))*np.ones((X_dim, (motif_width+2)*pool))
        seq_align[np.isnan(seq_align)] = 0
        W.append(seq_align)

    W = np.array(W)
    return W

def get_activate_W(model, hook_module, data, pool=1, threshold=0.99, motif_width=20):
    fmap, X = get_fmap(model, hook_module, data)
    W = get_activate_W_from_fmap(fmap, X, pool, threshold, motif_width)
    return W

def normalize_pwm(pwm, factor=None, max=None):
    if not max:
        max = np.max(np.abs(pwm))
    pwm = pwm/max
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    pwm = pwm/norm
    pwm[np.isnan(pwm)] = 0.25 # background
    return pwm

# pwm = W[0]
# pwm.shape == (4, 20)
# W.shape == (64, 4, 20)

def meme_generate(W, output_file='meme.txt', prefix='Motif_'):
    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('strands: + -\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j in range(len(W)):
        pwm = normalize_pwm(W[j])
        f.write('MOTIF %s%d %d\n' % (prefix, j, j))
        f.write('\n')
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d E= 0\n' % (pwm.shape[1], pwm.shape[1]))
        for i in range(pwm.shape[1]):
            f.write('  %.4f\t  %.4f\t  %.4f\t  %.4f\t\n' % tuple(pwm[:,i]))
        f.write('\n')

    f.close()

def filter_heatmap(pwm, output_fname=None, save=False, fig_size=(10, 7), 
                    norm=True, cmap='hot_r', cbar_norm=True):
    pwm_dim, pwm_len = pwm.shape

    plt.figure(figsize=fig_size)
    if norm:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    else:
        norm = None
    cmap_reversed = matplotlib.cm.get_cmap(cmap)
    im = plt.imshow(pwm, cmap=cmap_reversed, norm=norm, aspect="auto")

    #plt.axis('off')
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, pwm_len, 1.), minor=True)
    ax.set_yticks(np.arange(-.5, pwm_dim, 1.), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    plt.xticks(list(range(pwm_len)))
    if pwm.shape[0] == 4:
        plt.yticks([0, 1, 2, 3], ['A', 'C', 'G', 'T'], fontsize=16)
    else:
        plt.yticks(list(range(pwm_dim)), list(range(pwm_dim)), fontsize=16)

    #cbar = plt.colorbar()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    if cbar_norm:
        cbar.set_ticks([0.0, 0.5, 1.0])

    if save:
        plt.savefig(output_fname, format="pdf")
    plt.show()
    plt.close()

def filter_heatmap_W(W, factor=5, fig_size=(10,7), save=True):
    for idx, pwm in enumerate(W):
        output_fname = "Motif_" + str(idx) +".pdf"
        pwm = normalize_pwm(pwm, factor=factor)
        filter_heatmap(pwm, output_fname=output_fname, save=save, fig_size=fig_size)

def deep_explain_saliancy(model, input_tensor, n_class=1, use_abs=True):
    saliency = Saliency(model)
    saliency_val_l = []
    for i_class in range(n_class):
        attribution = saliency.attribute(input_tensor, target=i_class)
        saliency_vals = attribution.cpu().data.numpy()
        if use_abs:
            saliency_vals = np.abs(saliency_vals)
        saliency_val_l.append(saliency_vals)
    return np.array(saliency_val_l)

def input_saliancy_location(model, input_tensor, n_class=3, use_abs=True):
    saliency_val_l = deep_explain_saliancy(model, input_tensor, n_class=n_class, use_abs=use_abs)
    saliency_val = saliency_val_l.mean(0).mean(0).mean(0)
    saliency_length = pd.DataFrame(enumerate(saliency_val), columns=["location","saliancy"])
    return saliency_length

def plot_saliancy_location(model, input_tensor, n_class=3, use_abs=True):
    saliency_length = input_saliancy_location(model, input_tensor, n_class=n_class, use_abs=use_abs)
    plt.figure(figsize=(30,4))
    ax = sns.lineplot(x="location", y="saliancy", data=saliency_length)
    plt.show()
    plt.close()

def deep_explain_layer_conductance(model, model_layer, input_tensor, n_class=1):
    layer_cond = LayerConductance(model, model_layer)
    cond_val_l = []
    for i_class in range(n_class):
        attribution = layer_cond.attribute(input_tensor, target=i_class, internal_batch_size=32)
        cond_vals = attribution.detach().numpy()
        cond_val_l.append(cond_vals)
    return np.array(cond_val_l)

def label_neuron_importance(model, model_layer, input_tensor, label):
    n_class = len(label)
    imp = deep_explain_layer_conductance(model, model_layer, input_tensor, n_class=n_class)
    imp = imp.mean(-1).mean(1)
    df = pd.DataFrame(imp, index=label)
    return df

def plot_label_neuron_importance(model, model_layer, input_tensor, label):
    df = label_neuron_importance(model, model_layer, input_tensor, label)
    plt.figure(figsize=(30,4), cmap="vlag")
    ax = sns.heatmap(df)
    plt.show()
    plt.close()


def foldchange(origin, modified):
    return modified / origin
    # return np.square(modified - origin)

# hook
class ModifyOutputHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.channels = None
        self.channel = 0
    def hook_fn(self, module, input, output):
        for channel in self.channels:
            self.channel = channel
            if isinstance(module, torch.nn.modules.conv.Conv1d):
                output_channel = output[:,self.channel,:]
                output[:,self.channel,:] = torch.zeros_like(output_channel).to(output_channel.device)#output_channel.mean()
            elif isinstance(module, torch.nn.modules.linear.Linear):
                output_channel = output[:,self.channel]
                output[:,self.channel] = torch.zeros_like(output_channel).to(output_channel.device)#output_channel.mean()
            # logging.info(output_channel[:5].cpu().detach().numpy())
            # logging.info(output_channel.mean().cpu().detach().numpy())
        return output
    def step_channel(self, idx):
        if isinstance(idx, (list, tuple)):
            self.channels = idx
        elif isinstance(idx, int):
            self.channels = [idx]
    def get_current_channel(self):
        return self.channel
    def close(self):
        self.hook.remove()

def channel_target_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    criterion = torch.nn.BCELoss(reduction='none').to(device) # gene * cell
    target, pred_orig, loss_orig, pred_modified_foldchange = [], [], [], []

    model.eval()
    with torch.no_grad():
        for x_tensor, t in data_loader:
            x_tensor = x_tensor.to(device)
            t = t.to(device)
            output = model(x_tensor)
            loss = criterion(output, t)

            target.append(t.cpu().data.numpy())
            pred_orig.append(output.cpu().data.numpy())
            loss_orig.append(loss.cpu().data.numpy())
        target = np.vstack(target)
        pred_orig = np.vstack(pred_orig)
        loss_orig = np.vstack(loss_orig)
        logging.debug(pred_orig.shape)
        logging.debug(loss_orig.shape)
        logging.debug(pred_orig[:5,:5])
        logging.debug(loss_orig[:5,:5])

        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            out_channels = hook_module.out_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            out_channels = hook_module.out_features # must hook on linear layer
        Modifier = ModifyOutputHook(hook_module)
        for idx in range(out_channels):
            logging.info("modifying channel_%d..." % idx)
            pred_modified, loss_modified = [], []
            Modifier.step_channel(idx)
            for x_tensor, t in data_loader:
                x_tensor = x_tensor.to(device)
                t = t.to(device)
                output = model(x_tensor) # batch_size * output_size
                loss = criterion(output, t)

                pred_modified.append(output.cpu().data.numpy())
                loss_modified.append(loss.cpu().data.numpy())
            pred_modified = np.vstack(pred_modified) 
            loss_modified = np.vstack(loss_modified) 
            logging.debug(pred_modified.shape)
            logging.debug(loss_modified.shape)
            logging.debug(pred_modified[:5,:5])
            logging.debug(loss_modified[:5,:5])

            fc = foldchange(pred_orig, pred_modified).mean(0) # output_size
            # fc = foldchange(loss_orig, loss_modified).mean(0) # output_size
            pred_modified_foldchange.append(fc)
        Modifier.close()

    pred_modified_foldchange = np.vstack(pred_modified_foldchange)
    logging.debug(pred_modified_foldchange.shape)
    logging.debug(pred_modified_foldchange[:5,:5])

    return pred_modified_foldchange


def layer_channel_combination_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    pred_orig, pred_modified_foldchange = [], []

    model.eval()
    with torch.no_grad():
        for x_tensor, _ in data_loader:
            x_tensor = x_tensor.to(device)
            output = model(x_tensor).cpu().data.numpy()
            pred_orig.append(output)
        pred_orig = np.vstack(pred_orig)

        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            out_channels = hook_module.out_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            out_channels = hook_module.out_features # must hook on linear layer
        Modifier = ModifyOutputHook(hook_module)
        for idx in itertools.combinations(range(out_channels), 2):
            logging.info("modifying channel_%d&%d..." % idx)
            pred_modified = []
            Modifier.step_channel(idx)
            for x_tensor, _ in data_loader:
                x_tensor = x_tensor.to(device)
                output_modified = model(x_tensor).cpu().data.numpy() # batch_size * output_size
                pred_modified.append(output_modified)
            pred_modified = np.vstack(pred_modified) 
            fc = foldchange(pred_orig, pred_modified).mean(0) # output_size
            pred_modified_foldchange.append(fc)
        Modifier.close()

    return np.vstack(pred_modified_foldchange)


# hook
class ModifyInputHook():
    def __init__(self, module):
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.channels = None
        self.channel = 0
    def hook_fn(self, module, input):
        for channel in self.channels:
            self.channel = channel
            if isinstance(module, torch.nn.modules.conv.Conv1d):
                input_channel = input[0][:,self.channel,:]
                input[0][:,self.channel,:] = torch.zeros_like(input_channel).to(input_channel.device)#input_channel.mean()
            elif isinstance(module, torch.nn.modules.linear.Linear):
                input_channel = input[0][:,self.channel]
                input[0][:,self.channel] = torch.zeros_like(input_channel).to(input_channel.device)#input_channel.mean()
            # logging.info(input_channel[:5].cpu().detach().numpy())
            # logging.info(input_channel.mean().cpu().detach().numpy())
        return input
    def step_channel(self, idx):
        if isinstance(idx, (list, tuple)):
            self.channels = idx
        elif isinstance(idx, int):
            self.channels = [idx]
    def get_current_channel(self):
        return self.channel
    def close(self):
        self.hook.remove()


def input_channel_target_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    pred_orig, pred_modified_foldchange = [], []

    model.eval()
    with torch.no_grad():
        for x_tensor, _ in data_loader:
            x_tensor = x_tensor.to(device)
            output = model(x_tensor).cpu().data.numpy()
            pred_orig.append(output)
        pred_orig = np.vstack(pred_orig)

        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            in_channels = hook_module.in_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            in_channels = hook_module.in_features # must hook on linear layer
        Modifier = ModifyInputHook(hook_module)
        for idx in range(in_channels):
            logging.info("modifying channel_%d..." % idx)
            pred_modified = []
            Modifier.step_channel(idx)
            for x_tensor, _ in data_loader:
                x_tensor = x_tensor.to(device)
                output_modified = model(x_tensor).cpu().data.numpy() # batch_size * output_size
                pred_modified.append(output_modified)
            pred_modified = np.vstack(pred_modified) 
            fc = foldchange(pred_orig, pred_modified).mean(0)
            pred_modified_foldchange.append(fc)
        Modifier.close()

    return np.vstack(pred_modified_foldchange)


def input_layer_channel_combination_influence(model, hook_module, data_loader, device=torch.device("cuda")):
    pred_orig, pred_modified_foldchange = [], []

    model.eval()
    with torch.no_grad():
        for x_tensor, _ in data_loader:
            x_tensor = x_tensor.to(device)
            output = model(x_tensor).cpu().data.numpy()
            pred_orig.append(output)
        pred_orig = np.vstack(pred_orig)

        if isinstance(hook_module, torch.nn.modules.conv.Conv1d):
            in_channels = hook_module.in_channels # must hook on conv layer
        elif isinstance(hook_module, torch.nn.modules.linear.Linear):
            in_channels = hook_module.in_features # must hook on linear layer
        Modifier = ModifyInputHook(hook_module)
        for idx in itertools.combinations(range(in_channels), 2):
            logging.info("modifying channel_%d&%d..." % idx)
            pred_modified = []
            Modifier.step_channel(idx)
            for x_tensor, _ in data_loader:
                x_tensor = x_tensor.to(device)
                output_modified = model(x_tensor).cpu().data.numpy() # batch_size * output_size
                pred_modified.append(output_modified)
            pred_modified = np.vstack(pred_modified) 
            fc = foldchange(pred_orig, pred_modified).mean(0)
            pred_modified_foldchange.append(fc)
        Modifier.close()

    return np.vstack(pred_modified_foldchange)

