import pandas as pd
import numpy as np
import re

# species_fpath = "./Human_MAGIC_label_006_batchsize_96_20210930/Motif/tomtom_conv1_CisTarget_t1_allr/"
# motif_anno_fpath = "../0_TFBS/Motif_scenic/motifs-v9-nr.hgnc-m0.001-o0.0.tbl"

species_fpath = "./Mouse_MAGIC_label_006_batchsize_96_new/Motif/tomtom_conv1_CisTarget_t1_allr/"
motif_anno_fpath = "../0_TFBS/Motif_scenic/motifs-v9-nr.mgi-m0.001-o0.0.tbl"

# species_fpath = "./Dmel_MAGIC_label_015_batchsize_32/Motif/tomtom_conv1_CisTarget_t1_allr/"
# motif_anno_fpath = "../0_TFBS/Motif_scenic/motifs-v8-nr.flybase-m0.001-o0.0.tbl"

tomtom = pd.read_csv(species_fpath + "tomtom_conv1_CisTarget_t1_allr/tomtom.tsv", sep='\t', header=0, index_col=None, skipfooter=3, engine='python')
tomtom = tomtom[tomtom["q-value"]<0.1][["Query_ID", "Target_ID", "q-value"]]
tomtom


anno = pd.read_csv(motif_anno_fpath, sep='\t', header=0, index_col=None)[["#motif_id", "gene_name"]]
anno.columns = ["Motif_ID", "gene_name"]
anno.head()

def f(x):
    x = x.replace('__', '-')
    x = re.sub(r"swissregulon-.+-", "swissregulon-", x) # swissregulon
#     m = re.match(r"^taipale-(.+?)_(.+?)_(.+?)$")
    if x.find("taipale") != -1:
        s = x.split("-")[1].split("_")
        if len(s) > 3:
            x = '-'.join(["taipale", s[2], s[0], s[1]])
    return x
anno.index = anno.index.map(lambda x: f(x))
anno = anno[["motif_name", "gene_name"]]
anno

tomtom_anno = tomtom.merge(anno, how='inner', left_on="Target_ID", right_on="Motif_ID")
tomtom_anno

# tomtom_anno.columns = ["Target_ID", "q-value", "Query_ID", "Motif_ID", "gene_name"]
tomtom_anno.to_csv(species_fpath+"filter_anno.csv")

# anno = pd.read_csv("/media/ggj/Files/mount/NvWA_Final/TFBS/JASPAR2020/JASPAR.insects.anno.txt", sep=' ', header=None)
# anno.columns = ['MOTIF', 'ID', 'TF']

# tomtom = tomtom.merge(anno, left_on='Target_ID', right_on='ID', how='left')
# tomtom.to_csv("./tomtom_anno.tsv")