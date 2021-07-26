import os, time, shutil, logging, random, traceback, gc
from sys import argv
import pickle, h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from hyperopt import *
from utils import *

device_id, data = argv[1:]

os.makedirs('./Log', exist_ok=True)
os.makedirs('./hyperopt_trails', exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime('./Log/log_hyperopt.' + '%m%d.%H:%M:%S.txt'),
                    filemode='w')

# set random_seed
set_random_seed()
set_torch_benchmark()

## change
os.environ["CUDA_VISIBLE_DEVICES"] = device_id

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

species = os.path.basename(data).split('.')[1]
logging.info("#"*60)
logging.info("switching datasets: %s" % species)

# unpack datasets
h5file = h5py.File(data, 'r')
celltype = h5file["celltype"][:]
anno = h5file["annotation"][:]
anno = pd.DataFrame(anno, columns=["Cell", "Species", "Celltype", "Cluster"])
anno_cnt = anno.groupby("Celltype")["Species"].count()

x_train = h5file["train_data"][:].astype(np.float32)
y_train_onehot = h5file["train_label"][:].astype(np.float32)
logging.info(x_train.shape)
logging.info(y_train_onehot.shape)

x_val = h5file["val_data"][:].astype(np.float32)
y_val_onehot = h5file["val_label"][:].astype(np.float32)
logging.info(x_val.shape)
logging.info(y_val_onehot.shape)

h5file.close()

# define hyperparams
output_size = y_val_onehot.shape[-1]

batch_size = 16
EPOCH = 10

def run_trials(database):
    trials_step = 100  # how many additional trials to do after loading saved trials
    max_trials = 100 # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open(database, "rb"))
        logging.info("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        logging.info("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()
    
    best = fmin(objective, param_space, max_evals = max_trials, trials = trials,
        # algo = anneal.suggest)
        # algo = rand.suggest)
        algo = tpe.suggest)
        # algo = partial(mix.suggest, p_suggest=[(0.2, rand.suggest), (0.6, tpe.suggest), (0.2, anneal.suggest)]))

    # save the trials object
    with open(database, "wb") as f:
        pickle.dump(trials, f)

    logging.info(best)

def objective(params):
    global train_loader, validate_loader
    global model, optimizer, criterion
    global epoch
    
    params['is_spatial_transform'] = False
    params['anno_cnt'] = anno_cnt
    params['pred_prob'] = True
    params["output_size"] = output_size
    logging.info(params)
    
    # define dataset params
    leftpos = 10000 - int(params['pos'])
    rightpos = 10000 + int(params['pos'])
      
    # define train_datasets
    x_train_sub = x_train[:, :, leftpos:rightpos].copy()
    x_val_sub = x_val[:, :, leftpos:rightpos].copy()
    train_loader = DataLoader(list(zip(x_train_sub, y_train_onehot)), batch_size=batch_size,
                                shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    validate_loader = DataLoader(list(zip(x_val_sub, y_val_onehot)), batch_size=16, 
                            shuffle=False, num_workers=2, drop_last=False, pin_memory=True)
    
    logging.info(x_train_sub.shape)
    logging.debug('\n-----'+'0'+'------\n')
    logging.debug(params)

    try:
        model = get_model(params)
        logging.debug('\n-----'+'1'+'-------\n')
        model.to(device)
        logging.info(model.__str__())
        
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss().to(device)
        trainer = Trainer(model, criterion, optimizer, device)
        logging.debug('\n------'+'2'+'-------\n')
        logging.info('\n------'+'train'+'-------\n')

        val_loss_list = []
        for epoch in range(EPOCH):
            _ = trainer.train_per_epoch(train_loader, epoch, verbose_step=20)
            val_loss, _, _ = trainer.evaluate(validate_loader)
            val_loss_list.append(val_loss)
            logging.info("Eval\t Loss: %.4f\t\n" % val_loss)

        val_loss = np.min(val_loss_list)
        # logs
        logging.info("Eval\t Average Loss: %.4f\t\n" % val_loss)
        logging.debug('\n------'+'3'+'-------\n')
        
        del model, train_loader, validate_loader, x_train_sub, x_val_sub
        gc.collect()
        torch.cuda.empty_cache()
        
        return {'loss': val_loss, 'status': STATUS_OK }

    except:
        logging.warning(traceback.format_exc())
        logging.warning("-----model training Failed-----\n")
        return {'loss': 1, 'status': STATUS_FAIL } 


run_trials("./hyperopt_trails/params.p")