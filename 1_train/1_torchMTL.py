import os, time, shutil, logging, argparse
import pickle, h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from torchmtl import MTLModel
# import networkx as nx

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

class Flatten(nn.Module):
    def forward(self, fmap):
        fmap = fmap.view(fmap.size(0), -1) #flatten
        return fmap

# help func
class MTLoss(nn.Module):
    def __init__(self, lamda=1e-8):
        super().__init__()
        self.lamda = lamda
        if pred_prob:
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = nn.MSELoss()
    def forward(self, pred, target):
        L1_loss = 0
        for param in model.parameters():
            L1_loss += torch.sum(torch.abs(param))
        logging.debug(L1_loss)

        loss = self.loss_fn(pred, target)
        logging.debug(loss)

        MTloss = loss + self.lamda * L1_loss
        return MTloss

class Trainer(object):
    def __init__(self, model, criterion, optimizer, device):
        super().__init__()
        # train one epoch
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_per_epoch(self, train_loader, epoch, verbose_step=5):
        batch_losses = []
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Our model will return a list of predictions (from the layers specified in `output_tasks`),
            # loss functions, and regularization parameters (as defined in the tasks variable)
            y_hat, l_funcs, l_weights = self.model(data)

            loss = 0
            idx = 0
            # We can now iterate over the tasks and accumulate the losses
            for i, cnt in enumerate(anno_cnt.values):
                loss += l_weights[i] * l_funcs[i](y_hat[i], target[:,idx:idx+cnt])
                idx += cnt

            # self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            if verbose_step:
                if batch_idx % verbose_step == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data))
            batch_losses.append(loss.cpu().item())
            average_loss = np.average(batch_losses)

        return average_loss

    def evaluate(self, data):
        eval_loader = data
        batch_losses, all_predictions, all_targets = [], [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                y_hat, l_funcs, l_weights = self.model(inputs)

                loss = 0
                idx = 0
                # We can now iterate over the tasks and accumulate the losses
                for i, cnt in enumerate(anno_cnt.values):
                    loss += l_weights[i] * l_funcs[i](y_hat[i], targets[:,idx:idx+cnt])
                    idx += cnt

                test_loss = loss
                batch_losses.append(test_loss.cpu().item())

                output = torch.cat(y_hat, dim=1)
                all_predictions.append(output.cpu().data.numpy())
                all_targets.append(targets.cpu().data.numpy())
                
        average_loss = np.average(batch_losses)
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        return average_loss, all_predictions, all_targets

    def get_current_model(self):
        return self.model

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

# is_spatial_transform
is_spatial_transform = bool(params['is_spatial_transform'])

# define global avg pool
globalpoolsize = int(params['globalpoolsize'])
pooltype = params['pooltype']

# define conv params
numFiltersConv1 = int(params['numFiltersConv1'])
filterLenConv1 = int(params['filterLenConv1'])
# dilRate1 = int(params['dilRate1'])
isCBAM1 = params['CBAM1']
reduction_ratio1 = int(params['reduction_ratio1'])
maxPool1 = int(params['maxPool1'])
LSTM_input_size = numFiltersConv1
dense_input_size = numFiltersConv1 * globalpoolsize

# init CNN params
isLayer2 = numFiltersConv2 = filterLenConv2 = dilRate2 = isCBAM2 = reduction_ratio2 = maxPool2 = None
isLayer3 = numFiltersConv3 = filterLenConv3 = dilRate3 = isCBAM3 = reduction_ratio3 = maxPool3 = None

if params['numconvlayers']['numconvlayers1'] != 'one':
    isLayer2 = True,
    numFiltersConv2 = int(params['numconvlayers']['numFiltersConv2'])
    filterLenConv2 = int(params['numconvlayers']['filterLenConv2'])
    dilRate2 = int(params['numconvlayers']['dilRate2'])
    isCBAM2 = params['numconvlayers']['CBAM2']
    reduction_ratio2 = int(params['numconvlayers']['reduction_ratio2'])
    maxPool2 = int(params['numconvlayers']['maxPool2'])
    LSTM_input_size = numFiltersConv2
    dense_input_size = numFiltersConv2 * globalpoolsize
    
    if params['numconvlayers']['numconvlayers1']['numconvlayers2'] != 'two':
        isLayer3 = True,
        numFiltersConv3 = int(params['numconvlayers']['numconvlayers1']['numFiltersConv3'])
        filterLenConv3 = int(params['numconvlayers']['numconvlayers1']['filterLenConv3'])
        dilRate3 = int(params['numconvlayers']['numconvlayers1']['dilRate3'])
        isCBAM3 = params['numconvlayers']['numconvlayers1']['CBAM3']
        reduction_ratio3 = int(params['numconvlayers']['numconvlayers1']['reduction_ratio3'])
        maxPool3 = int(params['numconvlayers']['numconvlayers1']['maxPool3'])
        LSTM_input_size = numFiltersConv3
        dense_input_size = numFiltersConv3 * globalpoolsize

# init LSTM params
isRNN = LSTM_hidden_size = LSTM_hidden_layes = None
logging.debug(params['numRNNlayers'])
if params['numRNNlayers']['numRNNlayers1'] != 'zero':
    isRNN = True
    LSTM_hidden_size = int(params['numRNNlayers']['LSTM_hidden_size'])
    LSTM_hidden_layes = int(params['numRNNlayers']['LSTM_hidden_layes'])
    dense_input_size = LSTM_hidden_size * 2

# define dense params
dense1 = int(params['dense1'])
dropout1 = round(float(params['dropout1']), 2)
tower_input_size = dense1

# init FC params
is_fc2 = dense2 = dropout2 = None
is_fc3 = dense3 = dropout3 = None
if params['numdenselayers']['numdenselayers1'] != 'one':
    is_fc2 = True
    dense2 = int(params['numdenselayers']['dense2'])
    dropout2 = round(float(params['numdenselayers']['dropout2']), 2)
    tower_input_size = dense2
    
    if params['numdenselayers']['numdenselayers1']['numdenselayers2'] != 'two':
        is_fc3 = True
        dense3 = int(params['numdenselayers']['numdenselayers1']['dense3'])
        dropout3 = round(float(params['numdenselayers']['numdenselayers1']['dropout3']), 2)
        tower_input_size = dense3

# define dense params
tower_hidden = int(params['tower_hidden'])
tower_drop = round(float(params['tower_drop']))
        
Embedding = CNN(pooltype=pooltype, globalpoolsize=globalpoolsize,
                numFiltersConv1=numFiltersConv1, filterLenConv1=filterLenConv1, maxPool1=maxPool1,
                isCBAM1=isCBAM1, reduction_ratio1=reduction_ratio1, 
                isLayer2=isLayer2, numFiltersConv2=numFiltersConv2, filterLenConv2=filterLenConv2, dilRate2=dilRate2, maxPool2=maxPool2,
                isCBAM2=isCBAM2, reduction_ratio2=reduction_ratio2, 
                isLayer3=isLayer3, numFiltersConv3=numFiltersConv3, filterLenConv3=filterLenConv3, dilRate3=dilRate3, maxPool3=maxPool3,
                isCBAM3=isCBAM3, reduction_ratio3=reduction_ratio3, 
                is_spatial_transform=is_spatial_transform
                )

Mlp = MLP(dense_input_size=dense_input_size, dense1=dense1, dropout1=dropout1, 
            is_fc2=is_fc2, dense2=dense2, dropout2=dropout2, 
            is_fc3=is_fc3, dense3=dense3, dropout3=dropout3,)

flatten = Flatten()

tasks = [
    {
        'name': "Embedding",
        'layers': Embedding,
        # No anchor_layer means this layer receives input directly
    },
    {
        'name': "flatten",
        'layers': flatten,
        'anchor_layer': ['Embedding']
    },
    {
        'name': "MLP",
        'layers': Mlp,
        'anchor_layer': ['flatten']
    }
]

# branches for each task
output_tasks = []
for idx, cnt in anno_cnt.items():
    task_mlp = MLP(tower_input_size, tower_hidden, tower_drop, is_pred=True, pred_prob=pred_prob, output_size=cnt)
    task_name = "task_{}".format(idx)
    task_d = {
                'name': task_name, 
                'layers': task_mlp,
                'anchor_layer': ['MLP'],
                'loss': MTLoss(),
                'loss_weight': 1
    }
    tasks.append(task_d)
    output_tasks.append(task_name)

model = MTLModel(tasks, output_tasks=output_tasks)
model.Embedding.save(fname="./Log/embedder_init_params.pth")
logging.info("weights inited and embedding weights loaded")

# pos = nx.planar_layout(model.g)
# nx.draw(model.g, pos, font_size=14, node_color="y", node_size=450, with_labels=True)

model.to(device)
logging.info(model.__str__())

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
            torch.save(model, "./Log/best_model.pth")
            
        if epoch%20 == 1:
            torch.save(model, "./Log/model.pth")
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
    model = torch.load(fname+".params.pth")

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
    