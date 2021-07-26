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
parser.add_argument("--tower_by", dest="tower_by", default="Celltype")
parser.add_argument("--regression", dest="regression", action="store_true", default=False)
parser.add_argument("--use_data_rc_augment", dest="use_data_rc_augment", action="store_true", default=False)
parser.add_argument("--patience", dest="patience", default=5, type=int)
parser.add_argument("--lr", dest="lr", default=1e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--EPOCH", dest="EPOCH", default=100, type=int)

args = parser.parse_args()
data = args.data
device_id, mode, trails_fname = args.device_id, args.mode, args.trails # default
tower_by = args.tower_by
patience, lr, EPOCH = args.patience, args.lr, args.EPOCH
batch_size = args.batch_size
use_data_rc_augment = args.use_data_rc_augment
pred_prob = not args.regression

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
logging.info(args)

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


x_test = h5file["test_data"][:].astype(np.float32)
y_test_onehot = h5file["test_label"][:].astype(np.float32)

train_gene = h5file["train_gene"][:]
val_gene = h5file["val_gene"][:]
test_gene = h5file["test_gene"][:]

logging.info(x_test.shape)
logging.info(y_test_onehot.shape)

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
    logging.info("using model from trails:\t%s", trails_fname)

params['is_spatial_transform'] = False
params['anno_cnt'] = anno_cnt
params['pred_prob'] = pred_prob
logging.info(params)

# define datasets parameters
leftpos = int(params['leftpos'])
rightpos = int(params['rightpos'])
logging.info((leftpos, rightpos))

# define hyperparams
output_size = y_test_onehot.shape[-1]
params["output_size"] = output_size

model = get_model(params)
model.Embedding.save(fname="./Log/embedder_init_params.pth")
logging.info("weights inited and embedding weights loaded")

model.to(device)
logging.info(model.__str__())

class MTLoss(nn.Module):
    def __init__(self, lamda=1e-8):
        super().__init__()
        self.lamda = lamda
        self.L2loss = nn.MSELoss()
        if pred_prob:
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.MSELoss()
    def forward(self, pred, target):
        L1_loss = 0
        for param in model.parameters():
            L1_loss += torch.sum(torch.abs(param))
        logging.debug(L1_loss)
        
        within_loss_l = []
        c_mean_l, T_c_l = [], []
        for c in pred:
            logging.debug("c.shape")
            logging.debug(c.shape)
            T_c = c.shape[-1]
            T_c_l.append(T_c)

            c_mean = c.mean(1)
            logging.debug("c_mean.shape")
            logging.debug(c_mean.shape)
            c_mean_l.append(c_mean)
            for idx in range(T_c):
                p_t = c[:, idx]
                within_loss_l.append(self.L2loss(c_mean, p_t))
            logging.debug("p_t.shape")
            logging.debug(p_t.shape)
        within_loss = torch.mean(torch.stack(within_loss_l))
        logging.debug("within_loss")
        logging.debug(within_loss)

        pred = torch.cat(pred, dim=1)
        logging.debug("pred.shape")
        logging.debug(pred.shape)
        pred_mean = pred.mean(1)
        logging.debug("pred_mean.shape")
        logging.debug(pred_mean.shape)
        mean_loss = self.L2loss(torch.zeros_like(pred_mean).to(device), pred_mean)
        logging.debug("mean_loss")
        logging.debug(mean_loss)

        between_loss_l = []
        for c_mean, T_c in zip(c_mean_l, T_c_l):
            logging.debug("c_mean")
            logging.debug(c_mean.shape)
            logging.debug("T_c")
            logging.debug(T_c)

            between_loss = T_c * self.L2loss(pred_mean, c_mean)
            logging.debug("between_loss")
            logging.debug(between_loss)
            between_loss_l.append(between_loss)

        between_loss = torch.mean(torch.stack(between_loss_l))
        logging.debug(between_loss)

        var_loss = torch.var(target) - torch.var(pred)
        logging.debug(var_loss)

        loss = self.loss_fn(pred, target)
        logging.debug(loss)

        MTloss = loss + self.lamda * L1_loss + self.lamda * var_loss + 100 * self.lamda * within_loss + 10 * self.lamda * between_loss + self.lamda * mean_loss
        return MTloss

optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9)
criterion = MTLoss().to(device)
# criterion = nn.BCEWithLogitsLoss().to(device)
trainer = Trainer(model, criterion, optimizer, device)
logging.info('\n------'+'train'+'-------\n')

# define train_datasets
if mode != "test":
    x_train = x_train[:, :, leftpos:rightpos]
    x_val = x_val[:, :, leftpos:rightpos]
    logging.info(x_train.shape)
    logging.info(x_val.shape)

    if use_data_rc_augment:
        logging.info("using sequence reverse complement augment...")

        x_train, y_train_onehot = seq_rc_augment(x_train, y_train_onehot)
        logging.info(x_train.shape)

    train_loader = DataLoader(list(zip(x_train, y_train_onehot)), batch_size=batch_size,
                                shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    validate_loader = DataLoader(list(zip(x_val, y_val_onehot)), batch_size=16, 
                                shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

# test_loader
x_test = x_test[:, :, leftpos:rightpos]
logging.info(x_test.shape)

test_loader = DataLoader(list(zip(x_test, y_test_onehot)), batch_size=16, 
                            shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

if mode == "train" or mode == "resume":
    if mode == "resume":
        model.load_state_dict(torch.load("./Log/model.pth"))

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
        train_loss = trainer.train_per_epoch(train_loader, epoch, verbose_step=20)
        # _, train_pred_prob, train_target_prob = evaluate(train_loader)

        # fpr, tpr, roc_auc = calculate_roc(train_target_prob, train_pred_prob)
        # roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        # train_r = np.mean(roc_l)
        # show_auc_curve(fpr, tpr, roc_auc, output_fname='current_train_roc_curves.pdf')
        train_r = 0.5

        # validation
        val_loss, val_pred_prob, val_target_prob = trainer.evaluate(validate_loader)

        # fpr, tpr, roc_auc = calculate_roc(val_target_prob, val_pred_prob)
        # roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        # val_r = np.mean(roc_l)
        # show_auc_curve(fpr, tpr, roc_auc, output_fname='current_val_roc_curves.pdf')
        val_r = 0.5

        # lr_scheduler
        _lr = optimizer.param_groups[0]['lr']
        # lr_scheduler.step(val_loss)

        # test
        test_loss, test_pred_prob, test_target_prob = trainer.evaluate(test_loader)

        # fpr, tpr, roc_auc = calculate_roc(test_target_prob, test_pred_prob)
        # roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        # test_r = np.mean(roc_l)
        # show_auc_curve(fpr, tpr, roc_auc, output_fname='current_test_roc_curves.pdf')
        test_r = 0.5

        # logs
        logging.info("Train\t Accuracy: %.4f\t Loss: %.4f\t\n" % (train_r, train_loss))
        logging.info("Eval\t Accuracy: %.4f\t Loss: %.4f\t\n" % (val_r, val_loss))

        train_loss_list.append(train_loss); val_loss_list.append(val_loss); test_loss_list.append(test_loss)
        train_r_list.append(train_r); val_r_list.append(val_r); test_r_list.append(test_r)
        lrs.append(_lr) 
        show_train_log(train_loss_list, val_loss_list, val_r_list,
                test_loss_list, test_r_list, lrs, output_fname='current_logs.pdf')
        show_train_log(loss_train=train_loss_list, loss_val=val_loss_list, loss_test=test_loss_list, output_fname='current_logs_loss.pdf')
        show_train_log(loss_val=val_loss_list[-20:], output_fname='current_loss_val20.pdf')
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
            
        elif epoch%20 == 1:
            model.save("./Log/model.pth")
            logs = [train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs]
            pickle.dump(logs, open("./Log/chekc_train_log.p", 'wb'))

    show_train_log(loss_val=val_loss_list, output_fname='final_loss_val.pdf')
    show_train_log(loss_train=train_loss_list, output_fname='final_loss_train.pdf')
    show_train_log(acc_val=train_r_list, output_fname='final_acc_train.pdf')
    show_train_log(acc_val=val_r_list, output_fname='final_acc_val.pdf')
    show_train_log(acc_test=test_r_list, output_fname='final_acc_test.pdf')
    show_train_log(lrs=lrs, output_fname='final_lrs.pdf')

    current_r = np.max(test_r_list); current_mse = np.min(test_loss_list)
    logging.info(current_r, current_mse)

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
    os.makedirs("./Test", exist_ok=True)

    model.load_state_dict(torch.load("./Log/best_model.pth"))
    model.to(device)
    model.eval()

    test_target_l, test_pred_l, test_roc_l = [], [], []
    # test for 20 times
    for idx in range(2):
        # test
        logging.info("testing run %d ..." % idx)
        test_loss, test_pred_prob, test_target_prob = trainer.evaluate(test_loader)
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

    train_target_l, train_pred_l, train_roc_l = [], [], []
    val_target_l, val_pred_l, val_roc_l = [], [], []
    test_target_l, test_pred_l, test_roc_l = [], [], []
    # test for 20 times
    for idx in range(2):
        # test
        logging.info("testing run %d ..." % idx)
        train_loss, train_pred_prob, train_target_prob = trainer.evaluate(train_loader)
        logging.info(train_pred_prob[:5,:5])
        val_loss, val_pred_prob, val_target_prob = trainer.evaluate(validate_loader)
        logging.info(val_pred_prob[:5,:5])
        test_loss, test_pred_prob, test_target_prob = trainer.evaluate(test_loader)
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

    pd.DataFrame(np.vstack([train_target, val_target, test_target]), columns=celltype, index=np.hstack([train_gene, val_gene, test_gene])).to_pickle("./Test_All/test_target_prob.p", compression='xz')
    pd.DataFrame(np.vstack([train_pred, val_pred, test_pred]), columns=celltype, index=np.hstack([train_gene, val_gene, test_gene])).to_pickle("./Test_All/test_mode_pred_prob.p", compression='xz')
    logging.info("Prediction finished.")
  
    logging.info("Test All finished.")