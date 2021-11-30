import os, time, shutil, logging, argparse
import pickle, h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from sklearn.metrics import *
from hyperopt import *

from utils import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("data")
parser.add_argument("--mode", dest="mode", default="train")
parser.add_argument("--gpu-device", dest="device_id", default="0")
parser.add_argument("--trails", dest="trails", default="explainable")
parser.add_argument("--pos", dest="pos", default=None, type=int)
parser.add_argument("--globalpoolsize", dest="globalpoolsize", default=None, type=int)
parser.add_argument("--tower_hidden", dest="tower_hidden", default=None, type=int)
parser.add_argument("--tower_by", dest="tower_by", default="Celltype")
parser.add_argument("--regression", dest="regression", action="store_true", default=False)
parser.add_argument("--use_data_rc_augment", dest="use_data_rc_augment", action="store_true", default=False)
parser.add_argument("--patience", dest="patience", default=10, type=int)
parser.add_argument("--lr", dest="lr", default=1e-5, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=8, type=int)
parser.add_argument("--EPOCH", dest="EPOCH", default=500, type=int)

args = parser.parse_args()
data = args.data
device_id, mode = args.device_id, args.mode
trails_fname = args.trails # default
pos = args.pos
tower_by = args.tower_by
patience, lr, EPOCH = args.patience, args.lr, args.EPOCH
batch_size = args.batch_size
use_data_rc_augment = args.use_data_rc_augment
pred_prob = not args.regression

globalpoolsize = args.globalpoolsize
tower_hidden = args.tower_hidden

os.makedirs('./Log', exist_ok=True)
os.makedirs('./Figures', exist_ok=True)

# set random_seed
set_random_seed()
set_torch_benchmark()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime('./log_HyperBest.' + mode + '.%m%d.%H:%M:%S.txt'),
                    filemode='w')

## change
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
logging.info(device)

# unpack datasets
species = os.path.basename(data).split('.')[1]
logging.info("#"*60)
logging.info("switching datasets: %s" % species)

# unpack datasets
h5file = h5py.File(data, 'r')
celltype = h5file["celltype"][:]
anno = h5file["annotation"][:]
anno = pd.DataFrame(anno, columns=["Cell", "Species", "Celltype", "Cluster"])
anno_cnt = anno.groupby(tower_by, sort=False)["Species"].count()

if mode != "test":
    x_train = h5file["train_data"][:].astype(np.float32)
    y_train_onehot = h5file["train_label"][:].astype(np.float32)
    x_val = h5file["val_data"][:].astype(np.float32)
    y_val_onehot = h5file["val_label"][:].astype(np.float32)

    logging.info(x_train.shape)
    logging.info(x_val.shape)
    logging.info(y_train_onehot.shape)
    logging.info(y_val_onehot.shape)

    logging.debug(x_train[0,:,:5])
    logging.debug(y_train_onehot[0,:5])
    logging.debug(x_val[0,:,:5])
    logging.debug(y_val_onehot[0,:5])

x_test = h5file["test_data"][:].astype(np.float32)
y_test_onehot = h5file["test_label"][:].astype(np.float32)

logging.info(x_test.shape)
logging.info(y_test_onehot.shape)

logging.debug(x_test[0,:,:5])
logging.debug(y_test_onehot[0,:5])

train_gene = h5file["train_gene"][:]
val_gene = h5file["val_gene"][:]
test_gene = h5file["test_gene"][:]

logging.debug(train_gene[:5])
logging.debug(val_gene[:5])
logging.debug(test_gene[:5])

h5file.close()

# trails
if trails_fname == "best_manual":
    params = best_params
    logging.info("using best manual model")
elif trails_fname == "explainable":
    params = explainable_params
    logging.info("using explainable model")
elif trails_fname == "NIN":
    params = explainable_params
    params["is_NIN"] = True
    logging.info("using NIN model")
else:
    trials = pickle.load(open(trails_fname, 'rb'))
    best = trials.argmin
    params = space_eval(param_space, best)
    params['leftpos'] = 10000 - int(params['pos'])
    params['rightpos'] = 10000 + int(params['pos'])
    logging.info("using model from trails:\t%s", trails_fname)

params['is_spatial_transform'] = False
params['anno_cnt'] = anno_cnt
params['pred_prob'] = pred_prob

if pos:
    params['leftpos'] = 10000 - pos
    params['rightpos'] = 10000 + pos

# define datasets parameters
leftpos = int(params['leftpos'])
rightpos = int(params['rightpos'])
logging.info((leftpos, rightpos))

# define hyperparams
output_size = y_test_onehot.shape[-1]
params["output_size"] = output_size
if globalpoolsize:
    params['globalpoolsize'] = globalpoolsize
if tower_hidden:
    params['tower_hidden'] = tower_hidden

logging.info(params)

model = Beluga(rightpos-leftpos, output_size)
model.to(device)
logging.info(model.__str__())

optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
criterion = nn.BCELoss().to(device)
trainer = Trainer(model, criterion, optimizer, device)
logging.info('\n------'+'train'+'-------\n')

# define train_datasets
if mode != "test":
    x_train = x_train[:, :, leftpos:rightpos]
    x_val = x_val[:, :, leftpos:rightpos]
    logging.info(x_train.shape)
    logging.info(x_val.shape)
    logging.debug(x_train[0,:,:5])
    logging.debug(x_val[0,:,:5])

    if use_data_rc_augment:
        logging.info("using sequence reverse complement augment...")

        x_train, y_train_onehot = seq_rc_augment(x_train, y_train_onehot)
        logging.info(x_train.shape)

    train_loader = DataLoader(list(zip(x_train, y_train_onehot)), batch_size=batch_size,
                                shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    validate_loader = DataLoader(list(zip(x_val, y_val_onehot)), batch_size=batch_size, 
                                shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

# test_loader
x_test = x_test[:, :, leftpos:rightpos]
logging.info(x_test.shape)
logging.debug(x_test[0,:,:5])

test_loader = DataLoader(list(zip(x_test, y_test_onehot)), batch_size=batch_size, 
                            shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

if mode == "train" or mode == "resume":
    if mode == "resume":
        model.load_state_dict(torch.load("./Log/model.pth"))

    train_batch_loss_list, val_batch_loss_list, test_batch_loss_list = [], [], []
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    val_r_list, test_r_list, train_r_list = [], [], []
    lrs = []

    if mode == "resume":
        train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs = pickle.load(open("./Log/chekc_train_log.p", "rb"))
    elif mode == "train":
        logs = [train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs]
        pickle.dump(logs, open("./Log/chekc_train_log.p", 'wb'))

    _best_val_loss = np.inf
    _best_val_r = -np.inf
    _best_val_epoch = 0

    # train EPOCH
    for epoch in range(EPOCH):
        # early stop
        if epoch >= _best_val_epoch + patience: #patience_epoch:
            break

        # train
        train_batch_loss, train_loss = trainer.train_per_epoch(train_loader, epoch, verbose_step=20)
        # _, train_pred_prob, train_target_prob = evaluate(train_loader)

        # fpr, tpr, roc_auc = calculate_roc(train_target_prob, train_pred_prob)
        # roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        # train_r = np.mean(roc_l)
        # show_auc_curve(fpr, tpr, roc_auc, output_fname='current_train_roc_curves.pdf')
        train_r = 0.5

        # validation
        val_batch_loss, val_loss, val_pred_prob, val_target_prob = trainer.evaluate(validate_loader)

        # fpr, tpr, roc_auc = calculate_roc(val_target_prob, val_pred_prob)
        # roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        # val_r = np.mean(roc_l)
        # show_auc_curve(fpr, tpr, roc_auc, output_fname='current_val_roc_curves.pdf')
        val_r = 0.5

        # lr_scheduler
        _lr = optimizer.param_groups[0]['lr']
        # lr_scheduler.step(val_loss)

        # test
        test_batch_loss, test_loss, test_pred_prob, test_target_prob = trainer.evaluate(test_loader)

        # fpr, tpr, roc_auc = calculate_roc(test_target_prob, test_pred_prob)
        # roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        # test_r = np.mean(roc_l)
        # show_auc_curve(fpr, tpr, roc_auc, output_fname='current_test_roc_curves.pdf')
        test_r = 0.5

        # logs
        logging.info("Train\t Accuracy: %.4f\t Loss: %.4f\t\n" % (train_r, train_loss))
        logging.info("Eval\t Accuracy: %.4f\t Loss: %.4f\t\n" % (val_r, val_loss))

        train_batch_loss_list.extend(train_batch_loss); val_batch_loss_list.extend(val_batch_loss); test_batch_loss_list.extend(test_batch_loss)
        train_loss_list.append(train_loss); val_loss_list.append(val_loss); test_loss_list.append(test_loss)
        train_r_list.append(train_r); val_r_list.append(val_r); test_r_list.append(test_r)
        lrs.append(_lr) 
        show_train_log(train_loss_list, val_loss_list, val_r_list,
                test_loss_list, test_r_list, lrs, output_fname='current_logs.pdf')
        show_train_log(loss_train=train_loss_list, loss_val=val_loss_list, loss_test=test_loss_list, output_fname='current_logs_loss.pdf')
        show_train_log(loss_val=val_batch_loss_list, output_fname='current_batch_loss_val.pdf')
        show_train_log(loss_test=test_batch_loss_list, output_fname='current_batch_loss_test.pdf')
        show_train_log(loss_train=train_batch_loss_list, output_fname='current_batch_loss_train.pdf')
        show_train_log(loss_val=val_loss_list, output_fname='current_loss_val.pdf')
        show_train_log(loss_test=test_loss_list, output_fname='current_loss_test.pdf')
        show_train_log(loss_train=train_loss_list, output_fname='current_loss_train.pdf')
        show_train_log(acc_val=train_r_list, output_fname='current_acc_train.pdf')
        show_train_log(acc_val=val_r_list, output_fname='current_acc_val.pdf')
        show_train_log(acc_test=test_r_list, output_fname='current_acc_test.pdf')
        show_train_log(lrs=lrs, output_fname='current_lrs.pdf')

        # get_current_model
        model = trainer.get_current_model()
        # update best
        if val_loss < _best_val_loss or val_r > _best_val_r:
        # if val_r > _best_val_r:
            _best_val_loss = val_loss; _best_val_r = val_r; _best_val_epoch = epoch
            logging.info("Eval\t Best Val Accuracy: %.4f\t Loss: %.4f\t at Epoch: %d\t lr: %.8f\n" % (_best_val_r, _best_val_loss, epoch, _lr))
            logging.info("Eval\t Test Accuracy: %.4f\t Loss: %.4f\n" % (test_r, test_loss))
            show_train_log(train_loss_list, val_loss_list, val_r_list,
                test_loss_list, test_r_list, lrs,  output_fname='current_best_logs.pdf')
            # shutil.copyfile("./Figures/current_train_roc_curves.pdf", "./Figures/best_train_roc_curves.pdf")
            # shutil.copyfile("./Figures/current_val_roc_curves.pdf", "./Figures/best_val_roc_curves.pdf")
            # shutil.copyfile("./Figures/current_test_roc_curves.pdf", "./Figures/best_test_roc_curves.pdf")
            model.save("./Log/best_model.pth")
            # torch.save(model, "./Log/best_model.p")

        elif epoch%20 == 1:
            model.save("./Log/chekc_model.pth")
            # torch.save(model, "./Log/chekc_model.p")
            logs = [train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs]
            pickle.dump(logs, open("./Log/chekc_train_log.p", 'wb'))

    show_train_log(loss_val=val_loss_list, output_fname='final_loss_val.pdf')
    show_train_log(loss_train=train_loss_list, output_fname='final_loss_train.pdf')
    show_train_log(acc_val=train_r_list, output_fname='final_acc_train.pdf')
    show_train_log(acc_val=val_r_list, output_fname='final_acc_val.pdf')
    show_train_log(acc_test=test_r_list, output_fname='final_acc_test.pdf')
    show_train_log(lrs=lrs, output_fname='final_lrs.pdf')

    current_r = np.max(test_r_list); current_mse = np.min(test_loss_list)
    logging.info(current_r)
    logging.info(current_mse)

    best_NAS_mse = current_mse; best_NAS_r = current_r
    logging.info("\n"+model.__str__()+'\n'+"mse\t"+str(best_NAS_mse)+"\tR2_score\t"+str(best_NAS_r)+"\n")
    fname = time.strftime("./Log/best_model@" + '%m%d_%H:%M:%S')
    shutil.copyfile("./Log/best_model.pth", fname+".params.pth")
    model.load(fname+".params.pth")

    logs = [train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs]
    pickle.dump(logs, open(fname+".train_log.p", 'wb'))
    logging.info("update best NAS")
    logging.info(best_NAS_r)
    logging.info(best_NAS_mse)


elif mode == "test":
    os.makedirs(os.path.join("./Test"), exist_ok=True)

    model.load_state_dict(torch.load("./Log/best_model.pth"))
    model.to(device)
    model.eval()

    test_target_l, test_pred_l, test_roc_l = [], [], []
    # test for 20 times
    for idx in range(2):
        # test
        logging.info("testing run %d ..." % idx)
        _, _, test_pred_prob, test_target_prob = trainer.evaluate(test_loader)
        logging.info(test_target_prob[:5,:5])
        logging.info(test_pred_prob[:5,:5])

        test_target_l.append(test_target_prob)
        test_pred_l.append(test_pred_prob)
    
    test_target = np.array(test_target_l).mean(0).astype(np.float16)
    test_pred = np.array(test_pred_l).mean(0).astype(np.float16)
    
    logging.info(test_target.shape)
    logging.info(test_pred.shape)

    pd.DataFrame(test_target, columns=celltype, index=test_gene).to_pickle("./Test/test_target_prob.p", compression='xz')
    pd.DataFrame(test_pred, columns=celltype, index=test_gene).to_pickle("./Test/test_mode_pred_prob.p", compression='xz')
    logging.info("Prediction finished.")

    if pred_prob:
        fpr, tpr, roc_auc = calculate_roc(test_target, test_pred)
        roc_l = [roc_auc[k] for k in roc_auc.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
        # show_auc_curve(fpr, tpr, roc_auc, output_dir='./Test', output_fname='test_mode_roc_curves.pdf')
        pd.DataFrame(roc_l, index=celltype, columns=['AUROC_value']).to_csv("./Test/test_mode_roc.csv")
        logging.info(np.mean(roc_l))

        precision, recall, average_precision = calculate_pr(test_target, test_pred)
        ap_l = [average_precision[k] for k in average_precision.keys() if average_precision[k] >=0 and k not in ["macro", "micro"]]
        pd.DataFrame(ap_l, index=celltype, columns=['precision_value']).to_csv("./Test/test_mode_pr.csv")
        logging.info(np.mean(ap_l))

    else:
        correlation, pvalue = calculate_correlation(test_target, test_pred)
        correlation_l = [correlation[k] for k in correlation.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
        pvalue_l = [pvalue[k] for k in pvalue.keys() if k not in ["macro", "micro"]] 

        pd.DataFrame({"correlation":correlation_l, "pvalue":pvalue_l}, index=celltype).to_csv("./Test/test_mode_correlation.csv")
        logging.info(np.mean(correlation_l))
    
    logging.info("Test finished.")
    

elif mode == "test_all":
    os.makedirs("./Test_All", exist_ok=True)

    model.load_state_dict(torch.load("./Log/best_model.pth"))
    model.to(device)
    model.eval()

    # do not shuffle trainloader
    train_loader = DataLoader(list(zip(x_train, y_train_onehot)), batch_size=batch_size,
                                shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

    train_target_l, train_pred_l, train_roc_l = [], [], []
    val_target_l, val_pred_l, val_roc_l = [], [], []
    test_target_l, test_pred_l, test_roc_l = [], [], []
    # test for 20 times
    for idx in range(2):
        # test
        logging.info("testing run %d ..." % idx)
        _, _, train_pred_prob, train_target_prob = trainer.evaluate(train_loader)
        logging.info(train_pred_prob[:5,:5])
        _, _, val_pred_prob, val_target_prob = trainer.evaluate(validate_loader)
        logging.info(val_pred_prob[:5,:5])
        _, _, test_pred_prob, test_target_prob = trainer.evaluate(test_loader)
        logging.info(test_pred_prob[:5,:5])

        train_target_l.append(train_target_prob)
        train_pred_l.append(train_pred_prob)
        val_target_l.append(val_target_prob)
        val_pred_l.append(val_pred_prob)
        test_target_l.append(test_target_prob)
        test_pred_l.append(test_pred_prob)

    train_target = np.array(train_target_l).mean(0).astype(np.float16)
    train_pred = np.array(train_pred_l).mean(0).astype(np.float16)
    val_target = np.array(val_target_l).mean(0).astype(np.float16)
    val_pred = np.array(val_pred_l).mean(0).astype(np.float16)
    test_target = np.array(test_target_l).mean(0).astype(np.float16)
    test_pred = np.array(test_pred_l).mean(0).astype(np.float16)

    logging.info(test_target.shape)
    logging.info(test_pred.shape)

    target = np.vstack([train_target, val_target, test_target])
    pred = np.vstack([train_pred, val_pred, test_pred])
    genes = np.hstack([train_gene, val_gene, test_gene])

    logging.info(target.shape)
    logging.info(pred.shape)
    logging.info(target[:5,:5])
    logging.info(pred[:5,:5])

    pd.DataFrame(target, columns=celltype, index=genes).to_pickle("./Test_All/test_target_prob.p", compression='xz')
    pd.DataFrame(pred, columns=celltype, index=genes).to_pickle("./Test_All/test_mode_pred_prob.p", compression='xz')
    logging.info("Prediction finished.")

    if pred_prob:
        fpr, tpr, roc_auc = calculate_roc(target, pred)
        roc_l = [roc_auc[k] for k in roc_auc.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
        # show_auc_curve(fpr, tpr, roc_auc, output_dir='./Test', output_fname='test_mode_roc_curves.pdf')
        pd.DataFrame(roc_l, index=celltype, columns=['AUROC_value']).to_csv("./Test_All/test_mode_roc.csv")
        logging.info(np.mean(roc_l))

        precision, recall, average_precision = calculate_pr(target, pred)
        ap_l = [average_precision[k] for k in average_precision.keys() if average_precision[k] >=0 and k not in ["macro", "micro"]]
        pd.DataFrame(ap_l, index=celltype, columns=['precision_value']).to_csv("./Test_All/test_mode_pr.csv")
        logging.info(np.mean(ap_l))

    else:
        correlation, pvalue = calculate_correlation(target, pred)
        correlation_l = [correlation[k] for k in correlation.keys() if k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
        pvalue_l = [pvalue[k] for k in pvalue.keys() if k not in ["macro", "micro"]] 

        pd.DataFrame({"correlation":correlation_l, "pvalue":pvalue_l}, index=celltype).to_csv("./Test_All/test_mode_correlation.csv")
        logging.info(np.mean(correlation_l))

    logging.info("Test All finished.")