# NvWA
Code used for ```Deep learning of cross-species single-cell landscapes identifies conserved regulatory programs underlying cell types```

Nvwa, a deep learning–based strategy, to predict expression landscapes and decipher regulatory elements (Filters) at the single-cell level.

<div align=left><img src="./Logo.png" width="80px"></img></div>

## Requirements
- Python packages
```
h5py >= 2.7.0
numpy >= 1.14.2
pandas == 0.22.0
scipy >= 0.19.1
pyfasta >= 0.5.2
torch >= 1.0.0
captum
```

## Descriptions
- ```0_preproc_dataset``` for process dataset
- ```1_train``` for init, train and test models
- ```1_train/utils.py``` contains model architecture
- ```2_explain``` for explain models
- ```2_explain/explainer.py``` contains model explainer
- ```3_application``` for predicting genomic tracks
- ```main``` examples for run model in each species
- ```Analysis_plotting``` analysis and plotting function
- ```Results``` results of Nvwa analysis

#### Detailed description on Results Folder for Reproducing Figures
- ```Test_Metrics``` AUROC and AUPR Metric values on held-out test set for eight species
- ```scATAC_overlap_test``` Permutation test results of Nvwa whole-genome prediction and experimental functional genomics data
- ```Filters``` Property information of filters/motifs for eight species
- ```Filter_Annotation``` filters/motifs annotation results of TomTom agains known motif database
- ```Influe``` Influence scores (the fold-change of in-silico filter nullification on predictions)
- ```Influe_celltype``` detailed analysis of Influence scores
- ```Species_motif_hit.csv``` homologous Filters/motifs identified by TomTom among eight species
- ```tomtom_DBtfmodiscoTrimmed_NvwaConv1.html``` comparison of tfmodisco motifs and Nvwa featuremap-based motifs

For reproducing the Nvwa analysis from scratch, we recomand reading the `dmel.sh` in `main` folder, and downloading the drosophila dataset from the url below. 

## Datasets for eight species
We provided single cell labels for eight species in http://bis.zju.edu.cn/nvwa/dataset.html.

For the single cell labels, we provided the expression label, and corresponding cell, gene informations. The ready-to-use machine learning dataset were also publically accessed, which were paired with one-hot sequence, cell annotation information and split into train, validation, test set. The detailed preprocessing procedures were also described step by step.


## Running Nvwa
**Example**
```
python 1_train/1_hyperopt_BCE_best.py ./Dataset.Dmel_train_test.h5
python 1_train/1_hyperopt_BCE_best.py ./Dataset.Dmel_train_test.h5 --mode test
python 2_explain/1_run_explain.py ./Dataset.Dmel_train_test.h5
```
**Details**

`./Dataset.Dmel_train_test.h5`: example of Dataset.h5 file

`./1_train/1_hyperopt_BCE_best.py`: for init, train and test models

`--mode`: mode choice for train, test, test_all_gene

`2_explain/1_run_explain.py`: for explain models

`--help`: print help info.

## Note
Nvwa is now more like in-house scripts for reproducing our work, if you find any problem running Nvwa code, please contant me. If you run into errors loading trained model weights files, it is likely the result of differences in PyTorch or CUDA toolkit versions.  

NvTK (NvwaToolKit, https://github.com/JiaqiLiZju/NvTK), a more systemmatic software is under acitivate development. It will support modern deep learning achitectures in genomics, such as ResNet, Attention Module, and Transformer. I recommend to use NvTK for generating your own model.

## Citation
Please cite the corresponding protocol published concurrently to this repository:

Jiaqi Li, Jingjing Wang, Peijing Zhang, Renying Wang, Yuqing Mei, Zhongyi Sun, Lijiang Fei, Mengmeng Jiang,Lifeng Ma, Weigao E, Haide Chen,Xinru Wang, Yuting Fu, Hanyu Wu, Daiyuan Liu, Xueyi Wang, Jingyu Li, Qile Guo, Yuan Liao, Chengxuan Yu, Danmei Jia, Jian Wu, Shibo He, Huanju Liu, Jun Ma, Kai Lei, Jiming Chen, Xiaoping Han, Guoji Guo. Deep learning of cross-species single-cell landscapes identifies conserved regulatory programs underlying cell types. Nature Genetics, 2022. DOI: 10.1038/s41588-022-01197-7