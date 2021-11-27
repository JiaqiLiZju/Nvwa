#!/usr/bin/env python
# coding: utf-8

# # Run BLAST
file1 = 'data_pep/Human_Mouse/Human.fa'
type1 = 'prot' #or 'prot' if file1 is a proteome
id1 = 'hu' #2-character ID (e.g. 'hu' for human)

file2 = 'data_pep/Human_Mouse/Mouse_1.fa'
type2 = 'prot'
id2 = 'mo' #2-character ID (e.g. 'mo' for mouse)
!bash map_genes.sh --tr1 {file1} --t1 {type1} --n1 {id1} --tr2 {file2} --t2 {type2} --n2 {id2}


# # Run SAMap

from samap.mapping import SAMAP
from samap.analysis import get_mapping_scores, GenePairFinder, sankey_plot
from samalg import SAM

fn1 = '/home/jingjingw/Tool/samap_directory/data_h5ad/Human/Human.h5ad'
fn2 = '/home/jingjingw/Tool/samap_directory/data_h5ad/Mouse/Mouse.h5ad'
sm = SAMAP(fn1,fn2,id1,id2,f_maps = 'maps/')
samap = sm.run()
sm.samap = samap
sm
sm.sam1
sm.sam2
sm.samap
k1 = 'Celltype' #cell types annotation key in `sam1.adata.obs`
k2 = 'Celltype' #cell types annotation key in `sam2.adata.obs`
D1,D2,MappingTable = get_mapping_scores(sm,k1,k2, n_top = 0)
D1
D2
MappingTable.head()
MappingTable.to_csv("Result_SAMap_MappingTable_H_M.csv")
gpf = GenePairFinder(sm,k1=k1,k2=k2)
gene_pairs = gpf.find_all(thr=0.1)
gene_pairs.head()
gene_pairs.to_csv("Result_SAMap_gene_pairs_H_M.csv")

from samap.mapping import SAMAP
from samap.analysis import get_mapping_scores, GenePairFinder, sankey_plot
from samalg import SAM
from samap.utils import save_samap, load_samap
from samalg.gui import SAMGUI
sankey_plot(MappingTable)
import time
from datetime import datetime
now=datetime.now()
print(now)

