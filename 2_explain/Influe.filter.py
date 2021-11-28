import os, h5py
from sys import argv
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def norm_scale_influe(influe, clip_value=2):
    adata = sc.AnnData(influe.copy())

    sc.pp.normalize_total(adata, target_sum=10)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=np.inf)
    
    influe = adata.to_df().copy()
    if clip_value:
        influe = np.clip(influe, -clip_value, clip_value)

    return influe


def binary_influ(bin_influ_df, t=0.8):
    bin_influ_df = bin_influ_df.copy()

    t = np.quantile(bin_influ_df.values, t, axis=0)
    t[t < 0] = 0
    print(max(t), min(t))

    bin_influ_df[bin_influ_df <= t] = 0
    bin_influ_df[bin_influ_df > t] = 1
    bin_influ_df = bin_influ_df[bin_influ_df.sum(1) > 10]

    return bin_influ_df


def draw_influ_clustermap(bin_influ_df, anno, show_fig=True,
                          col_cluster=False, cmap='vlag', 
                          save_prefix="positive_influence", figtype='pdf',
                          save_dendrogram=False): 
    #import seaborn as sns
    #import matplotlib.pyplot as plt
    #from matplotlib.patches import Patch
    
    # clustermap
    color = ("#E6AB02", "#E41A1C", "#66A61E", "#D95F02", "#1B9E77", "#E7298A",  "#E31A1C", "#A6761D"  , "#B2DF8A",   "#FFFF99",   "#7570B3", "#FF7F00",  "#A65628", "#B3CDE3", "#BC80BD",     "#A6CEE3","#984EA3",   "#CCEBC5",  "#E41A1C",    "#4DAF4A","#BEBADA", "#B3DE69", "#CAB2D6","#FFFFB3",   "#33A02C","#B15928", "#6A3D9A","#FBB4AE",    "blue",          "#FB8072",      "#FFFF33","#CCEBC5",      "#A6761D",   "#2c7fb8","#fa9fb5",  "#BEBADA","#E7298A", "#E7298A", "green", "orange", "lightblue", "#BEBADA", "#33A02C", "#E31A1C", "#E6AB02", "#FFFF33", "lightblue", "#BC80BD", "#CCEBC5")
    regions = ("Secretory", "Germline", "Muscle", "Neuron" , "Immune", "Epithelial", "Glia", "Proliferating","Other",  "Neoblast","Protonephridia","Phagocytes","Cathepsin","Rectum", "Coelomocytes","Intestine","Hepatocyte","Pharynx","Endothelial","Erythroid","Testis","Mesenchyme","Yolk", "Midgut" ,"Embryo","Hemocytes",  "Fat",  "Unknown","Gastrodermis","DigFilaments","Pigment","BasementMembrane","Endoderm","RP_high","FatBody","Male","Nephron", "Pancreatic", "Neuroendocrine", "DigestiveGland", "Germ", "Stromal", "Non-seam", "Pharyn", "Precursors", "Seam", "Follicle", "MAG", "Notochord")
    color_regions = {x:y for x,y in zip(regions, color)}

    anno["colors_lineage"] = anno[['Cellcluster']].applymap(lambda x: color_regions[x])
    anno_color = anno.loc[bin_influ_df.index]
    
    lut = {cluster:color_regions.get(cluster) for cluster in anno.Cellcluster.unique()}
    handles = [Patch(facecolor=lut[name]) for name in lut]
    # print(lut)
    
    plt.figure(figsize=(6, 10))
    g = sns.clustermap(bin_influ_df.T, col_cluster=col_cluster,
                       cbar_pos=None, xticklabels=False,
                       col_colors=anno_color[["colors_lineage"]],
                       cmap=cmap, figsize=(18, 30),
                       dendrogram_ratio=(.01, .1), colors_ratio=0.01)

    plt.legend(handles, lut, title='CellLieange',
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

    plt.savefig(save_prefix+".clustermap."+figtype)
    if show_fig:
        plt.show()
    plt.close()
    
    if save_dendrogram and col_cluster:
        anno_color.iloc[g.dendrogram_col.reordered_ind,].to_csv(save_prefix+".cellanno.csv")
        print("saving dendrogram")


def melt_influ(bin_influ_df, Var="Celltype", t=None):
    bin_influ_df = bin_influ_df.copy()
    
    bin_influ_df['Var'] = bin_influ_df.index
    bin_influ_df = bin_influ_df.melt(id_vars='Var')
    bin_influ_df.columns = [Var, "Motif", "Influe"]
    
    if t:
        bin_influ_df = bin_influ_df[bin_influ_df.value > t]
        
    return bin_influ_df


def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta


def celltype_influ_analysis(influence, prefix):

    influence = norm_scale_influe(influence, clip_value=2)

    # anno_cell
    draw_influ_clustermap(influence, anno, show_fig=False, 
                          cmap='vlag', col_cluster=False, save_dendrogram=False,
                          save_prefix=prefix, figtype='png')
    draw_influ_clustermap(influence, anno, show_fig=False, 
                          cmap='vlag', col_cluster=True, save_dendrogram=True,
                          save_prefix=prefix+"_dendrogram", figtype='png')

    bin_influ = binary_influ(influence)
    draw_influ_clustermap(bin_influ, anno, show_fig=False, 
                            cmap='Greys', col_cluster=False, save_dendrogram=False,
                            save_prefix=prefix+"_bin", figtype='png')
    draw_influ_clustermap(bin_influ, anno, show_fig=False, 
                          cmap='Greys', col_cluster=True, save_dendrogram=True,
                          save_prefix=prefix+"_bin_dendrogram", figtype='png')

    # anno_celltype
    anno_celltype = anno_color.drop_duplicates(["Celltype", "Cellcluster"]).set_index("Celltype")
    influ_celltype = influence.groupby(anno_color.Celltype).mean()
    melt_influ(influ_celltype).to_csv(prefix+"_celltype.csv")

    draw_influ_clustermap(influ_celltype, anno_celltype, show_fig=False, 
                          cmap='vlag', col_cluster=False, save_dendrogram=False,
                          save_prefix=prefix+"_celltype", figtype='png')
    draw_influ_clustermap(influ_celltype, anno_celltype, show_fig=False, 
                          cmap='vlag', col_cluster=True, save_dendrogram=True,
                          save_prefix=prefix+"_celltype_dendrogram", figtype='png')


    bin_influ_celltype = binary_influ(influ_celltype)
    melt_influ(bin_influ_celltype).to_csv(prefix+"_celltype_bin.csv")
                    
    draw_influ_clustermap(bin_influ_celltype, anno_celltype, show_fig=False,
                            cmap='Greys', col_cluster=False, save_dendrogram=False,
                            save_prefix=prefix+"_celltype_bin", figtype='png')
    draw_influ_clustermap(bin_influ_celltype, anno_celltype, show_fig=False,
                          cmap='Greys', col_cluster=True, save_dendrogram=True,
                          save_prefix=prefix+"_celltype_bin_dendrogram", figtype='png')

    # anno_cluster
    anno_cluster = anno_color.drop_duplicates(["Cellcluster"])
    anno_cluster.index = anno_cluster.Cellcluster
    influ_cluster = influence.groupby(anno_color.Cellcluster).mean()
    melt_influ(influ_cluster, Var="Cluster").to_csv(prefix+"_cluster.csv")

    draw_influ_clustermap(influ_cluster, anno_cluster, show_fig=False,
                            cmap='vlag', col_cluster=True,
                            save_prefix=prefix+"_dendrogram")



os.makedirs("Influence/", exist_ok=True)

# dataset_fname = "../../../0_Dataset/Dataset.Smed_train_test.h5"
# anno_fname = "../../../0_Annotation_Cellcluster/Smed_GSE111764.cellatlas.annotation.20201215.txt"
dataset_fname, anno_fname, sample_size = argv[1:]
sample_size = int(sample_size)

# unpack datasets
h5file = h5py.File(dataset_fname, 'r')
cell_id = h5file["celltype"][:]
h5file.close()

anno = pd.read_csv(anno_fname, sep='\t', index_col=0, header=0).loc[cell_id]
# anno['Celltype'] = anno['Celltype'].apply(lambda x: x.split('_')[-1])
# anno["Celltype"] = anno.Celltype.apply(lambda x: '_'.join(x.split('_')[1:]) + '_' + x.split('_')[0])
# anno.head()

sel = (anno.Cellcluster != "Other") & (anno.Cellcluster != "RP_high") & (anno.Cellcluster != "Yolk")
anno = anno[sel]

if anno.shape[0] > sample_size:
    sample_mask = anno.sample(sample_size).index #
else:
    sample_mask = anno.index
anno_color = anno.loc[sample_mask]

## filter_influe
kernal_influence = 1 - pd.read_pickle("./influence_conv1.p", compression='xz').T.astype(float)[sel]
kernal_influence = kernal_influence.loc[sample_mask]
kernal_influence.columns = kernal_influence.columns.map(lambda x: 'Motif_' + str(x))

# influence_pos
influence_pos = kernal_influence.copy()
influence_pos[influence_pos < 0] = 0
influence_pos = influence_pos.loc[:,influence_pos.sum(0)!=0]
influence_pos = influence_pos.loc[anno.loc[influence_pos.index].sort_values(["Cellcluster", "Celltype"]).index]

prefix = "./Influence/positive_influence"
celltype_influ_analysis(influence_pos, prefix)

## comb_influe
comb_influe = 1 - pd.read_pickle("./influence_layer1_combination.p", compression='xz').T.astype(float)[sel]
comb_influe = comb_influe.loc[sample_mask]
comb_influe.iloc[:5,:4]

comb_influence_pos = comb_influe.copy()
comb_influence_pos[comb_influence_pos < 0] = 0
comb_influence_pos = comb_influence_pos.loc[:,comb_influence_pos.sum(0)!=0]
comb_influence_pos = comb_influence_pos.loc[anno.loc[comb_influence_pos.index].sort_values(["Cellcluster", "Celltype"]).index]

prefix = "./Influence/combination_positive_influence"
celltype_influ_analysis(comb_influence_pos, prefix)

## calculate eta correlation
eta_l = []
for i in range(kernal_influence.shape[-1]):
    eta = correlation_ratio(anno_color.Celltype, kernal_influence.iloc[:,i])
    eta_l.append(eta)

eta_comb_l = []
for i in range(comb_influe.shape[-1]):
    eta = correlation_ratio(anno_color.Celltype, comb_influe.iloc[:,i])
    eta_comb_l.append(eta)

data=pd.DataFrame({"type":np.hstack([["eta"]*kernal_influence.shape[-1], 
                                    ["eta_comb"]*comb_influe.shape[-1]]), 
                    "eta":np.hstack([eta_l, eta_comb_l])})
data.to_csv("./Influence/eta.csv")

plt.figure(figsize=(6, 10))
sns.boxplot(x="type", y="eta", data=data)
plt.savefig("./Influence/eta_boxplot.pdf")
