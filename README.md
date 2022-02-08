# NvWA
code used for ```Inferring genetic models from cross-species cell landscapes```

Nvwa, a deep learning–based strategy, to predict expression landscapes and decipher regulatory elements (Filters) at the single-cell level.

## Requirements
- Python packages
```
> h5py >= 2.7.0
> numpy >= 1.14.2
> pandas == 0.22.0
> scipy >= 0.19.1
> pyfasta >= 0.5.2
> torch >= 1.0.0
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

## Datasets for eight species
We provided single cell labels for eight species in URL.

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
Nvwa is now more like in-house scripts for reproducing our work in ```Inferring genetic models from cross-species cell landscapes```, if you find any problem running Nvwa code, please contant me.

NvTK (NvwaToolKit), a more systemmatic software is under acitivate development. It will support modern deep learning achitectures in genomics, such as ResNet, Attention Module, and Transformer. I recommend to use NvTK for generating your own model.
