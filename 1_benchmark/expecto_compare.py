import os, time, shutil, logging, random
import pickle, h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from hyperopt import *

from utils import *

import warnings
warnings.filterwarnings("ignore")

os.makedirs('./Log', exist_ok=True)
os.makedirs('./Figures', exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime('./log_Expecto.' + '%m%d.%H:%M:%S.txt'),
                    filemode='w')

# set random_seed
set_random_seed()
set_torch_benchmark()

from sys import argv
if len(argv)==4:
    logging.info("args init...")
    device_id, mode, data = argv[1:]
else:
    logging.warning("params mode is wrong")
    exit(1)

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
if mode != "test":
    x_train = h5file["train_data"][:].astype(np.float32)
    x_train =x_train.reshape(x_train.shape[0], -1)
    y_train = h5file["train_label"][:].astype(np.float32)

    x_val = h5file["val_data"][:].astype(np.float32)
    x_val = x_val.reshape(x_val.shape[0], -1)
    y_val = h5file["val_label"][:].astype(np.float32)

    logging.info(x_train.shape)
    logging.info(x_val.shape)

    logging.info(y_train.shape)
    logging.info(y_val.shape)

x_test = h5file["test_data"][:].astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = h5file["test_label"][:].astype(np.float32)
test_gene = h5file["test_gene"][:]

logging.info(x_test.shape)
logging.info(y_test.shape)

h5file.close()

# define hyperparams
output_size = y_test.shape[-1]
dense_input_size = x_test.shape[-1]

# define model
logging.debug('\n-----'+'0'+'------\n')
model = MLP(dense_input_size=dense_input_size, dense1=1024, dropout1=0.3, # use expecto features
            is_fc2=True, dense2=512, dropout2=0.3, 
            is_fc3=True, dense3=256, dropout3=0.3,
            is_pred=True, output_size=output_size)

model.initialize_weights()
# model.Embedding.save(fname="embedder_init_params.pth")
# model.Embedding.load_state_dict(torch.load("/share/home/guoguoji/JiaqiLi/nvwa/pretrain_deepsea/best_model@0708_05:06:04.params.pth"), strict=False)
logging.info("weights inited and embedding weights loaded")
logging.debug('\n-----'+'1'+'-------\n')

model.to(device)
logging.info(model.__str__())

lr = 1e-4
EPOCH = 500

optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9)

criterion = nn.BCELoss().to(device)
# criterion = nn.BCEWithLogitsLoss().to(device)
trainer = Trainer(model, criterion, optimizer, device)

logging.debug('\n------'+'2'+'-------\n')
logging.info('\n------'+'train'+'-------\n')

# define dataset params
batch_size = 32
# define train_datasets
if mode != "test":
    train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=batch_size,
                                shuffle=True, num_workers=0, drop_last=False)
    validate_loader = DataLoader(list(zip(x_val, y_val)), batch_size=batch_size, 
                                shuffle=False, num_workers=0, drop_last=False)
# test_loader
test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=batch_size, 
                            shuffle=False, num_workers=0, drop_last=False)

if mode == "train" or mode == "resume":
    if mode == "resume":
        model.load_state_dict(torch.load("model.pth"))

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
        if epoch >= _best_val_epoch + 5: #patience_epoch:
            break

        # train
        train_loss = trainer.train_per_epoch(train_loader, epoch, verbose_step=20)
        _, train_pred_prob, train_target_prob = trainer.evaluate(train_loader)

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
        
        # pd.DataFrame(test_target_prob, columns=celltype).to_csv("./Test/test_"+str(idx)+"_mode_target_prob.csv")
        # pd.DataFrame(test_pred_prob, columns=celltype, index=test_gene).to_csv("./Test/test_"+str(idx)+"_mode_pred_prob.csv")

        test_target_l.append(test_target_prob)
        test_pred_l.append(test_pred_prob)
    
    test_target = np.array(test_target_l).mean(0).astype(np.int8)
    test_pred = np.array(test_pred_l).mean(0).astype(np.float16)
    
    logging.info(test_target.shape)
    logging.info(test_pred.shape)

    fpr, tpr, roc_auc = calculate_roc(test_target_prob, test_pred_prob)
    roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]] # dict keys ordered by default in py3.7+
    test_r = np.mean(roc_l)
    # show_auc_curve(fpr, tpr, roc_auc, output_dir='./Test', output_fname='test_mode_roc_curves.pdf')

    pd.DataFrame(test_target, columns=celltype, index=test_gene).to_pickle("./Test/test_target_prob.p", compression='xz')
    pd.DataFrame(test_pred, columns=celltype, index=test_gene).to_pickle("./Test/test_mode_pred_prob.p", compression='xz')
    pd.DataFrame(roc_l, index=celltype, columns=['AUROC_value']).to_csv("./Test/test_mode_roc.csv")

    logging.info("Test finished.")
