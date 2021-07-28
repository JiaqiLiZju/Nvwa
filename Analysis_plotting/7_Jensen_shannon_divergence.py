#!/usr/bin/env python
# coding: utf-8

# # TF  jensen-shannon diverse
# 
# 
# __Author:__ Jingjing WANG
# 
# __Date:__ March 2021
# 
# __Six Species:__ Mouse_700gene


import time
from datetime import datetime
now=datetime.now()
print(now)

import numpy as np
import pandas as pd
import scanpy as sc

Adata_File = './Zebrafish_MAGIC.h5ad'
Meta_File = './Zebrafish_cellatlas.annotation.20210117.txt'
Meta_File2 = './Zebrafish_cellatlas.annotation.20210117_bionary.txt'
TF_File    = './Zebrafish_TF.txt'
OUT_File1  = 'TF_JSD.out'
OUT_File2  = 'TF_JSDR.out'

adata_SC = sc.read_h5ad(Adata_File)
adata_SC_df = adata_SC.to_df().T
adata_SC_df.columns = adata_SC_df.columns.astype("str")
adata_SC_df.index = adata_SC_df.index.astype("str")
adata_SC_df.shape

meta_SC = pd.read_table(Meta_File,sep="\t",header=0,index_col="Cell")
cell_index1 = adata_SC_df.columns.intersection(meta_SC.index)
cell_index1.shape[0]

import random
random.seed(10)
rc = random.sample(range (0,cell_index1.shape[0]),k= int(cell_index1.shape[0]*0.3))
cell_index = cell_index1[rc]

adata_SC_df = adata_SC_df.loc[:,cell_index]
meta_SC = meta_SC.loc [cell_index,:]

TF_S = pd.read_table(TF_File,sep="\t",header=0)
TF_S.shape

gene_index = adata_SC_df.index.intersection(TF_S['Symbol']) ##CHECK
gene_index.shape

adata_SC_df_use = adata_SC_df.loc[gene_index,:]

meta_SC2 = pd.read_table(Meta_File2,sep="\t")
meta_SC2 = meta_SC2.loc[:,cell_index]


Input1 = adata_SC_df_use # p matrix
Input2 = meta_SC2 # q matrix


Input1 = Input1.div(Input1.sum(axis=1),axis='rows')
Input2 = Input2.div(Input2.sum(axis=1),axis='rows')

JSD_list = []
JSDR_list = []

import math
for i in range(Input1.shape[0]):
    for j in range(Input2.shape[0]):
        p = np.asarray(Input1.iloc[i,:]) + np.spacing(1)
        q = np.asarray(Input2.iloc[j,:]) + np.spacing(1)
        M=(p+q)/2 + np.spacing(1)   
        jsd_value=0.5*np.sum(p*np.log2(p/M))+0.5*np.sum(q*np.log2(q/M))
        jsd_r = 1- math.sqrt(jsd_value)
        JSD_list.append([Input1.index[i],Input2.index[j],jsd_value])
        JSDR_list.append([Input1.index[i],Input2.index[j],jsd_r])


JSD_pd = pd.DataFrame(JSD_list,columns=['Cell','Celltype','JSD'])
JSDR_pd = pd.DataFrame(JSDR_list,columns=['Cell','Celltype','JSDR'])

JSD_pd.to_csv(OUT_File1,index = 0,sep="\t")
JSDR_pd.to_csv(OUT_File2,index = 0,sep="\t")


