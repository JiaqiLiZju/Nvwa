import os, time, shutil, logging, argparse
import pickle, h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from hyperopt import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("dataDir")
parser.add_argument("--mode", dest="mode", default="train")
parser.add_argument("--trails", dest="trails", default="best_manual")
parser.add_argument("--gpu-device", dest="device_id", default="0")
parser.add_argument("--use_data_rc_augment", dest="use_data_rc_augment", action="store_true", default=False)

args = parser.parse_args()
dataDir = args.dataDir
device_id, mode, trails_fname, use_data_rc_augment =  args.device_id, args.mode, args.trails, args.use_data_rc_augment # default False
# device_id, mode, trails_fname = "0", "train", "../nvwa-pse-official/hyper_params_tune/TPE-lr5-epoch10/hyperopt_trails/params.p"

os.makedirs('./Log', exist_ok=True)
os.makedirs('./Figures', exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime('./Log/log_pretrain_deepsea.' + mode + '.%m%d.%H:%M:%S.txt'),
                    filemode='w')

set_random_seed()
set_torch_benchmark()

device = torch.device("cuda" if torch.cuda.is_available() and device_id else "cpu")
logging.info(device)

# unpack datasets
def load_datasets(fname, mode="train"):
    try:
        import scipy.io as scio

        data = scio.loadmat(fname)
        X = data[mode+'xdata']
        y = data[mode+'data']
    except:
        data = h5py.File(fname, 'r')
        X = data[mode+'xdata'][:].swapaxes(0, -1)
        y = data[mode+'data'][:].swapaxes(0, -1)
        data.close()
        
    if X.shape[-1] == 4:
        X = X.swapaxes(1, -1)
    if X.dtype != np.float:
        X = X.astype(np.float32)
    if y.dtype != np.float:
        y = y.astype(np.float32)

    logging.info(mode)
    logging.info(X.shape)
    logging.info(y.shape)
    
    return X, y

# load datasets
if mode != "test":
    x_train, y_train = load_datasets(os.path.join(dataDir, "train.mat"), mode="train")
    x_val, y_val = load_datasets(os.path.join(dataDir, "valid.mat"), mode="valid")

    y_train_onehot = y_train
    y_val_onehot = y_val

    logging.info(x_train.shape)
    logging.info(x_val.shape)
    logging.info(y_train_onehot.shape)
    logging.info(y_val_onehot.shape)

x_test, y_test = load_datasets(os.path.join(dataDir, "test.mat"), mode="test")
y_test_onehot = y_test

logging.info(x_test.shape)
logging.info(y_test_onehot.shape)

# trails and params
if trails_fname == "best_manual":
    params = best_params
    logging.info(params)
else:
    trials = pickle.load(open(trails_fname, 'rb'))
    best = trials.argmin
    params = space_eval(params, best)
    logging.info(params)

# define dataset params
batch_size = int(params['batchsize'])
# leftpos = int(params['leftpos'])
# rightpos = int(params['rightpos'])
# logging.info((leftpos, rightpos))

# define hyperparams
output_size = y_test_onehot.shape[-1]
params['output_size'] = output_size

params['is_spatial_transform'] = False

model = get_model(params)
model.Embedding.save(fname="./Log/embedder_init_params.pth")
# model.Embedding.load_state_dict(torch.load("/share/home/guoguoji/JiaqiLi/nvwa/pretrain_deepsea/best_model@0708_05:06:04.params.pth"), strict=False)
logging.info("weights inited and embedding weights loaded")

model.to(device)
logging.info(model.__str__())

lr = 1e-3
EPOCH = 100

optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,)
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9)
criterion = nn.BCELoss().to(device)
# criterion = nn.BCEWithLogitsLoss().to(device)
trainer = Trainer(model, criterion, optimizer, device)

logging.debug('\n------'+'2'+'-------\n')
logging.info('\n------'+'train'+'-------\n')

# define train_datasets
if mode != "test":
#     x_train = x_train[:, :, leftpos:rightpos]
#     x_val = x_val[:, :, leftpos:rightpos]
    logging.info(x_train.shape)
    logging.info(x_val.shape)

    if use_data_rc_augment:
        logging.info("using sequence reverse complement augment...")

        x_train, y_train_onehot = seq_rc_augment(x_train, y_train_onehot)
        logging.info(x_train.shape)

    train_loader = DataLoader(list(zip(x_train, y_train_onehot)), batch_size=batch_size,
                                shuffle=True, num_workers=0, drop_last=False)
    validate_loader = DataLoader(list(zip(x_val, y_val_onehot)), batch_size=32, 
                                shuffle=False, num_workers=0, drop_last=False)

# test_loader
# x_test = x_test[:, :, leftpos:rightpos]
logging.info(x_test.shape)

test_loader = DataLoader(list(zip(x_test, y_test_onehot)), batch_size=32, 
                            shuffle=False, num_workers=0, drop_last=False)

if mode == "train" or mode == "resume":
    if mode == "resume":
        model.load_state_dict(torch.load("./Log/model.pth"))

    train_loss_list, val_loss_list, test_loss_list = [], [], []
    val_r_list, test_r_list, train_r_list = [], [], []
    lrs = []

    if mode == "resume":
       train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs = pickle.load(open("./log/check_train_log.p", "rb"))
    elif mode == "train":
        logs = [train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs]
        pickle.dump(logs, open("./Log/check_train_log.p", 'wb'))

    _best_val_loss = np.inf
    _best_val_r = -np.inf
    _best_val_epoch = 0

    # train EPOCH
    for epoch in range(EPOCH):
        # early stop
        if epoch >= _best_val_epoch + 10: #patience_epoch:
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

        fpr, tpr, roc_auc = calculate_roc(val_target_prob, val_pred_prob)
        roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        val_r = np.mean(roc_l)
        show_auc_curve(fpr, tpr, roc_auc, output_fname='current_val_roc_curves.pdf')
        
        # lr_scheduler
        _lr = optimizer.param_groups[0]['lr']
        # lr_scheduler.step(val_loss)

        # test
        test_loss, test_pred_prob, test_target_prob = trainer.evaluate(test_loader)
        
        fpr, tpr, roc_auc = calculate_roc(test_target_prob, test_pred_prob)
        roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        test_r = np.mean(roc_l)
        show_auc_curve(fpr, tpr, roc_auc, output_fname='current_test_roc_curves.pdf')
        
        # logs
        logging.info("Train\t Accuracy: %.4f\t Loss: %.4f\t\n" % (train_r, train_loss))
        logging.info("Eval\t Accuracy: %.4f\t Loss: %.4f\t\n" % (val_r, val_loss))

        train_loss_list.append(train_loss); val_loss_list.append(val_loss); test_loss_list.append(test_loss)
        train_r_list.append(train_r); val_r_list.append(val_r); test_r_list.append(test_r)
        lrs.append(_lr) 
        show_train_log(train_loss_list, val_loss_list, val_r_list,
                test_loss_list, test_r_list, lrs, output_fname='current_logs.pdf')
        show_train_log(loss_val=val_loss_list[-20:], output_fname='current_loss_val20.pdf')
        show_train_log(loss_val=val_loss_list, output_fname='current_loss_val.pdf')
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
            shutil.copyfile("./Figures/current_val_roc_curves.pdf", "./Figures/best_val_roc_curves.pdf")
            shutil.copyfile("./Figures/current_test_roc_curves.pdf", "./Figures/best_test_roc_curves.pdf")
            model.save("best_model.pth")
            
        elif epoch%20 == 1:
            model.save("model.pth")
            logs = [train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs]
            pickle.dump(logs, open("chekc_train_log.p", 'wb'))

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
    shutil.copyfile("best_model.pth", fname+".params.pth")
    model.load(fname+".params.pth")

    logs = [train_loss_list, val_loss_list, test_loss_list, train_r_list, val_r_list, test_r_list, lrs]
    pickle.dump(logs, open(fname+".train_log.p", 'wb'))
    logging.info("update best NAS")
    logging.info(best_NAS_r)
    logging.info(best_NAS_mse)

elif mode == "test":
    os.makedirs("./Test", exist_ok=True)

    model.load_state_dict(torch.load("./best_model.pth"))
    model.to(device)
    model.eval()

    test_target_l, test_pred_l, test_roc_l = [], [], []
    # test for 20 times
    for idx in range(20):
        # test
        all_predictions, all_targets  = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                all_predictions.append(output.cpu().data.numpy())
                all_targets.append(targets.cpu().data.numpy())
        test_pred_prob = np.vstack(all_predictions)
        test_target_prob = np.vstack(all_targets)

        fpr, tpr, roc_auc = calculate_roc(test_target_prob, test_pred_prob)
        roc_l = [roc_auc[k] for k in roc_auc.keys() if roc_auc[k] >=0 and k not in ["macro", "micro"]]
        test_r = np.mean(roc_l)
        show_auc_curve(fpr, tpr, roc_auc, output_dir='./Test', output_fname='test_'+str(idx)+'_mode_roc_curves.pdf')

        pd.DataFrame(test_pred_prob).to_csv("./Test/test_"+str(idx)+"_mode_pred_prob.csv")
        #, columns=celltype, index=test_gene

        test_target_l.append(test_target_prob)
        test_pred_l.append(test_pred_prob)
        test_roc_l.append(roc_l)
    
    test_target = np.array(test_target_l).mean(0)
    test_pred = np.array(test_pred_l).mean(0)
    test_roc = np.array(test_roc_l).mean(0)
    
    logging.info(test_target.shape)
    logging.info(test_roc.shape)

    pd.DataFrame(test_target).to_csv("./Test/test_target_prob.csv")
    pd.DataFrame(test_pred).to_csv("./Test/test_mode_pred_prob.csv")
    pd.DataFrame(test_roc).to_csv("./Test/test_mode_roc.csv")

    logging.info("Test finished.")