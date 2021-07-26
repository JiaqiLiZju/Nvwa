import numpy as np
import pandas as pd
import os, shutil
from sys import argv

from utils import calculate_roc, calculate_pr, calculate_correlation

prefix = argv[1]

target_fname = os.path.join(prefix+"1", "Test/test_target_prob.p")
print(target_fname)
assert os.path.isfile(target_fname)
target_df = pd.read_pickle(target_fname, compression='xz')
test_target = target_df.values
celltype = target_df.columns
test_gene = target_df.index

pred_l = []
for i in range(1, 11):
    pred_fname = os.path.join(prefix+str(i), "Test/test_mode_pred_prob.p")
    print(pred_fname)
    assert os.path.isfile(pred_fname)
    pred_l.append(pd.read_pickle(pred_fname, compression='xz').values)

test_pred = np.array(pred_l).mean(0)

shutil.copy(target_fname, "./test_target_prob_cv.p")
pd.DataFrame(test_pred, columns=celltype, index=test_gene).to_pickle("./test_mode_pred_prob_cv.p", compression='xz')

# test metrics
# fpr, tpr, roc_auc = calculate_roc(test_target, test_pred)
# roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
# print(np.mean(roc_l))
# pd.DataFrame(roc_l, index=celltype, columns=['AUROC_value']).to_csv("./test_mode_roc_cv.csv")

# precision, recall, average_precision = calculate_pr(test_target, test_pred)
# ap_l = [average_precision[k] for k in average_precision.keys() if average_precision[k] >=0 and k not in ["macro", "micro"]]
# print(np.mean(ap_l))
# pd.DataFrame(ap_l, index=celltype, columns=['precision_value']).to_csv("./test_mode_pr_cv.csv")

correlation, pvalue = calculate_correlation(test_target, test_pred)
correlation_l = [correlation[k] for k in correlation.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
pvalue_l = [pvalue[k] for k in pvalue.keys() if k not in ["macro", "micro"]] 
pd.DataFrame({"correlation":correlation_l, "pvalue":pvalue_l}, index=celltype).to_csv("./test_mode_correlation.csv")
print(np.mean(correlation_l))

# pred_l, target_l, test_gene_l = [], [], []
# for i in range(1, 11):
#     target_fname = os.path.join(prefix+str(i), "Test/test_target_prob.p")
#     print(target_fname)
#     assert os.path.isfile(target_fname)
#     target_df = pd.read_pickle(target_fname, compression='xz')
#     celltype = target_df.columns.values
#     test_gene = target_df.index.values
#     test_gene_l.append(test_gene)

#     test_target = target_df.values
#     target_l.append(test_target)

#     pred_fname = os.path.join(prefix+str(i), "Test/test_mode_pred_prob.p")
#     print(pred_fname)
#     assert os.path.isfile(pred_fname)
#     pred_l.append(pd.read_pickle(pred_fname, compression='xz').values)

# test_pred = np.vstack(pred_l)
# test_target = np.vstack(target_l)
# print(test_pred.shape)
# print(test_target.shape)

# test_gene = np.hstack(test_gene_l)
# print(test_gene.shape)

# fpr, tpr, roc_auc = calculate_roc(test_target, test_pred)
# roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
# test_r = np.mean(roc_l)

# pd.DataFrame(test_target, columns=celltype, index=test_gene).to_pickle("./test_target_prob_cv.p", compression='xz')
# pd.DataFrame(test_pred, columns=celltype, index=test_gene).to_pickle("./test_mode_pred_prob_cv.p", compression='xz')
# pd.DataFrame(roc_l, index=celltype, columns=['AUROC_value']).to_csv("./test_mode_roc_cv.csv")