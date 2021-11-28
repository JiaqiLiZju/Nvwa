import re
import numpy as np
import pandas as pd

tomtom = pd.read_csv("./tomtom.tsv", sep='\t', header=0, index_col=0, skipfooter=3, engine='python')
tomtom = tomtom[tomtom["q-value"]<0.1][["Target_ID", "q-value"]]

anno = pd.read_csv("/media/ggj/Files/mount/NvWA_Final/TFBS/Dmel_scenic/motifs-v8-nr.flybase-m0.001-o0.0.tbl", 
                  sep='\t', header=0, index_col=0)

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

# anno.source_name.unique()
# anno[anno.source_name=='yetfasco'].index.unique()
# anno.index.intersection(tomtom.Target_ID)

tomtom_anno = tomtom.merge(anno, how='left', left_on="Target_ID", right_index=True)
tomtom_anno[tomtom_anno.motif_name.isna()].Target_ID.unique()

tomtom_anno = tomtom_anno.dropna()
tomtom_anno.to_csv("./filter_anno.csv")

d = {}
for _, gene in tomtom_anno[["Query_ID", "gene_name"]].iterrows():
    m = gene.Query_ID
    tf = gene.gene_name
    if m not in d.keys():
        d[m] = [tf]
    else:
        tfs = d[m]
        if tf not in tfs:
            d[m].append(tf)

for k,v in d.items():
    d[k] = '; '.join(v)

filter_tf_anno = pd.DataFrame(d, index=['tfs']).T
filter_tf_anno.to_csv("./filter_tf_anno.csv")