import h5py
from sys import argv

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from utils import calculate_roc

data = argv[1]

# unpack datasets
h5file = h5py.File(data, 'r')
celltype = h5file["celltype"][:]
x_train = h5file["train_data"][:].astype(np.float32)
y_train = h5file["train_label"][:].astype(np.float32)

x_test = h5file["test_data"][:].astype(np.float32)
y_test = h5file["test_label"][:].astype(np.float32)
test_gene = h5file["test_gene"][:]

# y_train, y_test = y_train[:,:10], y_test[:,:10]

print(x_train.shape)
print(y_train.shape)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', max_iter=-1)) # iter untill converge

y_score = classifier.fit(x_train, y_train).decision_function(x_test)
print(y_score.shape)

fpr, tpr, roc_auc = calculate_roc(y_test, y_score)
roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
print(np.mean(roc_l))

pd.DataFrame(y_test, columns=celltype, index=test_gene).to_pickle("test_target_prob.p", compression='xz')
pd.DataFrame(y_score, columns=celltype, index=test_gene).to_pickle("test_mode_pred_prob.p", compression='xz')
pd.DataFrame(roc_l, index=celltype, columns=['AUROC_value']).to_csv("test_mode_roc.csv")
print("finished.")