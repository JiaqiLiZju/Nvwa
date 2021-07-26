import os, time, shutil, logging, pickle, h5py, random, itertools, math
import numpy as np
import pandas as pd
from numpy import interp
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr, kendalltau

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import sigmoid, softmax

from hyperopt import *
from sklearn.metrics import *

import networkx as nx
from copy import copy # only used in logging.DEBUG mode

def set_random_seed(random_seed = 12):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)

def set_torch_benchmark():
    # set torch benchmark
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seq_rc_augment(X, y):
    X_rc = np.array([seq[::-1,::-1] for seq in X]) # reverse_comp
    y_rc = y # keep same label
    return np.vstack((X_rc, X)), np.vstack((y_rc, y))

def onehot_encode(label):
    from sklearn.preprocessing import label_binarize
    return label_binarize(label, classes=range(np.max(label)+1))

def map_prob2label(y_pred_prob, map_fn=np.argmax):
    assert isinstance(y_pred_prob, np.ndarray)
    return np.array(list(map(map_fn, y_pred_prob)))

param_space = {
    # datasets params
    'pos' : hp.quniform('pos', 0, 10000, 500), #[0, 10000, 500]
    # 'batchsize' : 2**hp.quniform('batchsize', 4, 7, 1), #16-128
    # 'leftpos' : hp.quniform('leftpos', 0, 10000, 500), #[0, 10000, 500]
    # 'rightpos' : hp.quniform('rightpos', 10000, 20000, 500), #[10000, 20000, 500]
    
    # conv params
    'pooltype' : hp.choice('pooltype', ['avgpool', 'maxpool']),

    'numFiltersConv1' : 2**hp.quniform('numFiltersConv1', 4, 10, 1), #16-128
    'filterLenConv1' : hp.quniform('filterLenConv1', 4, 20, 1), #1-10
    # 'dilRate1' : hp.quniform('dilRate1', 1, 2, 1), # dilRate1 = 1
    'CBAM1' : hp.choice('CBAM1', [True, False]),
    'reduction_ratio1' : 2**hp.quniform('reduction_ratio1', 1, 5, 1), #2-16
    'maxPool1' : hp.quniform('maxPool1', 4, 20, 1), #[1,10,1]
    'numconvlayers' : hp.choice('numconvlayers', [
    {
        'numconvlayers1' : 'one'
    },
    {
        'numFiltersConv2' : 2**hp.quniform('numFiltersConv2', 4, 10, 1), #16-256
        'filterLenConv2' : hp.quniform('filterLenConv2', 4, 20, 1),
        'dilRate2' : hp.quniform('dilRate2', 1, 4, 1),
        'CBAM2' : hp.choice('CBAM2', [True, False]),
        'reduction_ratio2' : 2**hp.quniform('reduction_ratio2', 1, 5, 1), #2-16
        'maxPool2' : hp.quniform('maxPool2', 4, 20, 1),
        'numconvlayers1' : hp.choice('numconvlayers1', [
        {
            'numconvlayers2' : 'two'
        },
        {
            'numFiltersConv3' : 2**hp.quniform('numFiltersConv3', 4, 10, 1), #16-512
            'filterLenConv3' : hp.quniform('filterLenConv3', 4, 20, 1),
            'dilRate3' : hp.quniform('dilRate3', 1, 4, 1),
            'CBAM3' : hp.choice('CBAM3', [True, False]),
            'reduction_ratio3' : 2**hp.quniform('reduction_ratio3', 1, 5, 1), #2-16
            'maxPool3' : hp.quniform('maxPool3', 4, 20, 1),
            'numconvlayers2' : 'three'
        }])
    }]),
    
    'globalpoolsize' : 2**hp.quniform('globalpoolsize', 5, 11, 1), #64-1024
    
    # RNN params
    'numRNNlayers': hp.choice('numRNNlayers', [
        {
            'numRNNlayers1' : 'zero'
        },
        {
            'LSTM_hidden_size' : 2**hp.quniform('LSTM_hidden_size', 4, 10, 1), #16-1,024*2
            'LSTM_hidden_layes' : hp.quniform('LSTM_hidden_layes', 1, 3, 1), #16-256
            'numRNNlayers1' : 'one'
        }
    ]),
    
    # FC params
    'dense1' : 2**hp.quniform('dense1', 1, 16, 1), #1,024-32,768
    'dropout1' : hp.uniform('dropout1', 0, 1),
    'numdenselayers' : hp.choice('numdenselayers', [
    {
        'numdenselayers1' : 'one'
    },
    {
        'dense2' : 2**hp.quniform('dense2', 1, 14, 1), #1,024-8,192
        'dropout2' : hp.uniform('dropout2', 0, 1),
        'numdenselayers1' : hp.choice('numdenselayers1', [
        {
            'numdenselayers2' : 'two'
        },
        {
            'dense3' : 2**hp.quniform('dense3', 1, 12, 1), #1,024-4,096
            'dropout3' : hp.uniform('dropout3', 0, 1),
            'numdenselayers2' : 'three' ,
        }])
    }]),
    
    # TOWER params
    'tower_hidden': 2**hp.quniform('tower_hidden', 1, 8, 1), #1,024-4,096
    'tower_drop' : hp.uniform('tower_drop', 0, 1),

}

best_params = {
    # datasets params
    'batchsize' : 128, 
    'leftpos' : 3500,
    'rightpos' : 13500,
    
    # conv params
    'pooltype' : 'avgpool',

    'numFiltersConv1' : 64,
    'filterLenConv1' : 7,
    # 'dilRate1' : hp.quniform('dilRate1', 1, 2, 1), # dilRate1 = 1
    'CBAM1': False,
    'reduction_ratio1' : 16,
    'maxPool1' : 7,
    'numconvlayers': {
        'numFiltersConv2' : 128,
        'filterLenConv2' : 7,
        'dilRate2' : 2,
        'CBAM2' : False,
        'reduction_ratio2': 16,
        'maxPool2' : 7,
        'numconvlayers1' : {
            'numFiltersConv3' : 512,
            'filterLenConv3' : 9,
            'dilRate3' : 3,
            'CBAM3' : False,
            'reduction_ratio3' : 16,
            'maxPool3' : 9,
            'numconvlayers2' : 'three'
        }
    },
    
    'globalpoolsize' : 8,
    
    # RNN params
    'numRNNlayers': {
            'numRNNlayers1' : 'zero'},
    
    # FC params
    'dense1' : 2048,
    'dropout1' : 0.7,
    'numdenselayers' : {
        'dense2' :2048, 
        'dropout2' : 0.5,
        'numdenselayers1' : {
            'dense3' : 2048,
            'dropout3' : 0.3,
            'numdenselayers2' : 'three' ,
        }
    },

    # TOWER params
    'tower_hidden': 32,
    'tower_drop' : 0.1,
}

explainable_params = {
    # datasets params
    'batchsize' : 32, 
    'leftpos' : 3500,
    'rightpos' : 16500,
    
    # conv params
    'pooltype' : 'avgpool',

    'numFiltersConv1' : 128,
    'filterLenConv1' : 7,
    # 'dilRate1' : hp.quniform('dilRate1', 1, 2, 1), # dilRate1 = 1
    'CBAM1': False,
    'reduction_ratio1' : 16,
    'maxPool1' : 7,
    'numconvlayers': {
        'numFiltersConv2' : 256,
        'filterLenConv2' : 7,
        'dilRate2' : 2,
        'CBAM2' : False,
        'reduction_ratio2': 16,
        'maxPool2' : 7,
        'numconvlayers1' : {'numconvlayers2' : 'two'}
    },
    
    'globalpoolsize' : 64,
    
    # RNN params
    'numRNNlayers': {
            'numRNNlayers1' : 'zero'},
    
    # FC params
    'dense1' : 256,
    'dropout1' : 0.3,
    'numdenselayers' : {
        'dense2' : 256, 
        'dropout2' : 0.3,
        'numdenselayers1' : 'one'#{'numdenselayers2' : 'two'}
    },
    
    # TOWER params
    'tower_hidden': 32,
    'tower_drop' : 0.1,
}

def get_model(params):
    global anno_cnt, output_size
    # define hyperparams
    output_size = int(params['output_size'])
    anno_cnt = params["anno_cnt"]
    pred_prob = params.get("pred_prob", True) # default True

    if params.get("is_NIN", False):
        model = NIN(output_size)
        return model

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

    logging.debug('\n-----'+'0'+'------\n')
    model = Net(pooltype=pooltype, globalpoolsize=globalpoolsize,
                numFiltersConv1=numFiltersConv1, filterLenConv1=filterLenConv1, maxPool1=maxPool1,
                isCBAM1=isCBAM1, reduction_ratio1=reduction_ratio1, 
                isLayer2=isLayer2, numFiltersConv2=numFiltersConv2, filterLenConv2=filterLenConv2, dilRate2=dilRate2, maxPool2=maxPool2,
                isCBAM2=isCBAM2, reduction_ratio2=reduction_ratio2, 
                isLayer3=isLayer3, numFiltersConv3=numFiltersConv3, filterLenConv3=filterLenConv3, dilRate3=dilRate3, maxPool3=maxPool3,
                isCBAM3=isCBAM3, reduction_ratio3=reduction_ratio3, is_spatial_transform=is_spatial_transform,
                isRNN=isRNN, LSTM_input_size=LSTM_input_size, LSTM_hidden_size=LSTM_hidden_size, LSTM_hidden_layes=LSTM_hidden_layes,
                dense_input_size=dense_input_size, dense1=dense1, dropout1=dropout1, 
                is_fc2=is_fc2, dense2=dense2, dropout2=dropout2, 
                is_fc3=is_fc3, dense3=dense3, dropout3=dropout3,
                tower_input_size=tower_input_size, tower_hidden=tower_hidden, tower_drop=tower_drop,
                output_size=output_size, pred_prob=pred_prob)

    logging.debug('\n-----'+'1'+'-------\n')
    model.initialize_weights()
    return model


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight) 
                if m.bias is not None:
                    m.bias.data.zero_()
                logging.debug("init conv/linear")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                logging.debug("init BatchNorm1d")
            elif isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.all_weights[0][0])
                nn.init.orthogonal_(m.all_weights[0][1])
                nn.init.orthogonal_(m.all_weights[1][0])
                nn.init.orthogonal_(m.all_weights[1][1])
                logging.debug("init LSTM")
    def initialize_weights_from_pretrained(self, pretrained_net_fname):
        pretrained_dict = torch.load(pretrained_net_fname)
        net_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict)
        self.load_state_dict(net_state_dict)
        logging.debug("params loaded")
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    def save(self, fname=None):
        if fname is None:
            fname = time.strftime("model" + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), fname)
        return fname


class DeepSEA(BasicModule):
    def __init__(self, sequence_length, n_genomic_features):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            # nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            # nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(),
            # nn.Dropout(p=0.5)
            )
        
        self.n_channels = 128
        self.GlobalAvgPool = nn.AdaptiveAvgPool1d(self.n_channels)
        
#         reduce_by = conv_kernel_size - 1
#         pool_kernel_size = float(pool_kernel_size)
#         self.n_channels = int(
#             np.floor(
#                 (np.floor(
#                     (sequence_length - reduce_by) / pool_kernel_size)
#                  - reduce_by) / pool_kernel_size)
#             - reduce_by)
#         self.classifier = nn.Sequential(
#             nn.Linear(960 * self.n_channels, n_genomic_features),
#             nn.ReLU(inplace=True),
#             nn.Linear(n_genomic_features, n_genomic_features),
#             nn.Sigmoid())
#             )

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = self.conv_net(x)
        out = self.GlobalAvgPool(out)
#         out = out.view(out.size(0), 960 * self.n_channels)
#         predict = self.classifier(reshape_out)
        return out


class Beluga1D(BasicModule):
    def __init__(self, is_clf=True, sequence_length=2000, n_genomic_features=2002):
        super(Beluga1D, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.is_clf = is_clf

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(),
            
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.Dropout(p=0.2)
            )

        # self.n_channels = 107
        # self.GlobalAvgPool = nn.AdaptiveAvgPool1d(self.n_channels)

        if self.is_clf:
            self.classifier = nn.Sequential(
                # nn.Linear(960 * self.n_channels, n_genomic_features),
                nn.Linear(960*107, n_genomic_features),
                nn.ReLU(inplace=True),
                nn.Linear(n_genomic_features, n_genomic_features),
                nn.Sigmoid()
                )

    def forward(self, x):
        logging.debug(x.shape)

        out = self.conv_net(x)
        logging.debug(out.shape) # N * channels(960) * sequence_length_after_pooling(107)

        if self.is_clf:
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            logging.debug(out.shape) # N * n_genomic_features(2002)

        return out


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class CNN(BasicModule):
    def __init__(self, pooltype = 'maxpool', globalpoolsize=128,
                    numFiltersConv1=128, filterLenConv1=3, maxPool1=3,
                    isCBAM1=False, reduction_ratio1=16, 
                    isLayer2=True, numFiltersConv2=256, filterLenConv2=3, dilRate2=2, maxPool2=3,
                    isCBAM2=False, reduction_ratio2=16, 
                    isLayer3=True, numFiltersConv3=512, filterLenConv3=3, dilRate3=2, maxPool3=3,
                    isCBAM3=False, reduction_ratio3=16, 
                    is_spatial_transform=False
                    ):
        
        super().__init__()
        if pooltype == 'avgpool':
            poollayer = nn.AvgPool1d
        else:
            poollayer = nn.MaxPool1d
        
        # define conv1
        conv1_layers = OrderedDict()
        conv1_layers["conv"] = nn.Conv1d(in_channels=4, out_channels=numFiltersConv1, kernel_size=filterLenConv1, bias=True)
        # conv1_layers["bn"] = nn.BatchNorm1d(numFiltersConv1)
        if isCBAM1:
            conv1_layers["cbam"] = CBAM(gate_channels=numFiltersConv1, reduction_ratio=reduction_ratio1, no_spatial=False)
        conv1_layers["activate"] = nn.ReLU()
        conv1_layers["pool"] = poollayer(maxPool1)
        self.conv1 = nn.Sequential(conv1_layers)

        # define conv2
        self.isLayer2 = isLayer2
        if self.isLayer2:
            conv2_layers = OrderedDict()
            conv2_layers["conv"] = nn.Conv1d(in_channels=numFiltersConv1, out_channels=numFiltersConv2, kernel_size=filterLenConv2, dilation=dilRate2, bias=True)
            # conv2_layers["bn"] = nn.BatchNorm1d(numFiltersConv2)
            if isCBAM2:
                conv2_layers["cbam"] = CBAM(gate_channels=numFiltersConv2, reduction_ratio=reduction_ratio2, no_spatial=False)
            conv2_layers["activate"] = nn.ReLU()
            conv2_layers["pool"] = poollayer(maxPool2)
            self.conv2 = nn.Sequential(conv2_layers)

        # define conv3
        self.isLayer3 = isLayer3
        if self.isLayer3:
            conv3_layers = OrderedDict()
            conv3_layers["conv"] = nn.Conv1d(in_channels=numFiltersConv2, out_channels=numFiltersConv3, kernel_size=filterLenConv3, dilation=dilRate3, bias=True)
            # conv3_layers["bn"] = nn.BatchNorm1d(numFiltersConv3)
            if isCBAM3:
                conv3_layers["cbam"] = CBAM(gate_channels=numFiltersConv3, reduction_ratio=reduction_ratio3, no_spatial=False)
            conv3_layers["activate"] = nn.ReLU()
            conv3_layers["pool"] = poollayer(maxPool3)
            self.conv3 = nn.Sequential(conv3_layers)

        # define globalpooling
        self.globalpoolsize = globalpoolsize
        self.globalpooling = nn.AdaptiveAvgPool1d(globalpoolsize)
        
        # is_spatial_transform
        self.is_spatial_transform = is_spatial_transform # must use cuda if true
        self.pos_weights = None

    def forward(self, x):
        logging.debug(x.shape)

        out = self.conv1(x)
        logging.debug(out.shape)

        if self.isLayer2:
            out = self.conv2(out)
            logging.debug(out.shape)

        if self.isLayer2 and self.isLayer3:
            out = self.conv3(out)
            logging.debug(out.shape)

        # sequence_length = out.size(-1)
        # if self.globalpoolsize < sequence_length:
        out = self.globalpooling(out) # batchsize * filtersize * globalpoolsize
        logging.debug(out.shape)
        
        if self.is_spatial_transform:
            out = self.spatial_transformation(out)

        return out

    def spatial_transformation(self, x, bin=8):
        if self.pos_weights is None:
            bin_width = int(x.size(-1) / bin)
            TSS_pos = int(x.size(-1) / 2)
            shifts = np.array(list(range(-TSS_pos, TSS_pos, 1)))
            pos_weights = np.vstack([
                np.exp(-0.01 * np.abs(shifts) / bin_width) * (shifts <= 0),
                np.exp(-0.02 * np.abs(shifts) / bin_width) * (shifts <= 0),
                np.exp(-0.05 * np.abs(shifts) / bin_width) * (shifts <= 0),
                np.exp(-0.1 * np.abs(shifts) / bin_width) * (shifts <= 0),
                np.exp(-0.2 * np.abs(shifts) / bin_width) * (shifts <= 0),
                np.exp(-0.01 * np.abs(shifts) / bin_width) * (shifts >= 0),
                np.exp(-0.02 * np.abs(shifts) / bin_width) * (shifts >= 0),
                np.exp(-0.05 * np.abs(shifts) / bin_width) * (shifts >= 0),
                np.exp(-0.1 * np.abs(shifts) / bin_width) * (shifts >= 0),
                np.exp(-0.2 * np.abs(shifts) / bin_width) * (shifts >= 0)
                ])
            self.pos_weights = torch.from_numpy(pos_weights.sum(axis=0).astype(np.float32)).cuda() # [10, 200]

        x_new = self.pos_weights[None,None,:] * x[:,:,:]
        return x_new

class RNN(nn.Module):
    def __init__(self, LSTM_input_size=512, LSTM_hidden_size=512, LSTM_hidden_layes=2):
        super(RNN, self).__init__()
        self.rnn_hidden_state = None
        self.rnn = nn.LSTM(
            input_size=LSTM_input_size,
            hidden_size=LSTM_hidden_size,
            num_layers=LSTM_hidden_layes,
            batch_first=True,  # batch, seq, feature
            bidirectional=True,
        )
    def forward(self, input):
        output, self.rnn_hidden_state = self.rnn(input, None)
        logging.debug(output.shape)

        return output


class MLP(BasicModule):
    def __init__(self, dense_input_size=1024, dense1=512, dropout1=0.5, 
                 is_fc2=False, dense2=512, dropout2=0.5, 
                 is_fc3=False, dense3=256, dropout3=0.5,
                 is_pred=False, pred_prob=True, output_size=1):
        super(MLP, self).__init__()
        self.is_fc2 = is_fc2
        self.is_fc3 = is_fc3
        self.is_pred = is_pred
        
        self.fc1 = nn.Sequential(
            nn.Linear(dense_input_size, dense1),
            # nn.BatchNorm1d(dense1),
            nn.ReLU(),
            nn.Dropout(dropout1),
        )
        pred_input = dense1

        if self.is_fc2:
            self.fc2 = nn.Sequential(
                nn.Linear(dense1, dense2),
                # nn.BatchNorm1d(dense2),
                nn.ReLU(),
                nn.Dropout(dropout2),
            )
            pred_input = dense2

        if self.is_fc2 and self.is_fc3:
            self.fc3 = nn.Sequential(
                nn.Linear(dense2, dense3),
                # nn.BatchNorm1d(dense3),
                nn.ReLU(),
                nn.Dropout(dropout3),
            )
            pred_input = dense3

        if self.is_pred:
            if pred_prob:
                self.pred = nn.Sequential(
                    nn.Linear(pred_input, output_size),
                    nn.Sigmoid()
                )
            else:
                self.pred = nn.Sequential(
                    nn.Linear(pred_input, output_size)
                )
        
    def forward(self, fmap):
        out = self.fc1(fmap)
        logging.debug(out.shape)

        if self.is_fc2:
            out = self.fc2(out)
            logging.debug(out.shape)

        if self.is_fc2 and self.is_fc3:
            out = self.fc3(out)
            logging.debug(out.shape)
        
        if self.is_pred:
            out = self.pred(out)
            logging.debug(out.shape)
            
        return out

class ConcatenateTask(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=1)

class Net(BasicModule):
    def __init__(self, isCNN=True, pooltype = 'maxpool', globalpoolsize=128,
                    numFiltersConv1=128, filterLenConv1=3, maxPool1=3,
                    isCBAM1=False, reduction_ratio1=16, 
                    isLayer2=True, numFiltersConv2=256, filterLenConv2=3, dilRate2=2, maxPool2=3,
                    isCBAM2=False, reduction_ratio2=16, 
                    isLayer3=True, numFiltersConv3=512, filterLenConv3=3, dilRate3=2, maxPool3=3,
                    isCBAM3=False, reduction_ratio3=16, 
                    is_spatial_transform = False,
                 
                    isRNN=None, LSTM_input_size=128, LSTM_hidden_size=512, LSTM_hidden_layes=2,

                    isDense=True,
                    dense_input_size=1024, dense1=512, dropout1=0.5, 
                    is_fc2=None, dense2=512, dropout2=0.5, 
                    is_fc3=None, dense3=256, dropout3=0.5,

                    tower_input_size=512, tower_hidden=32, tower_drop=0.3,
                    output_size=7, pred_prob=True):
        
        super(Net, self).__init__()

        self.isCNN = isCNN
        if self.isCNN:
            self.Embedding = CNN(pooltype=pooltype, globalpoolsize=globalpoolsize,
                                numFiltersConv1=numFiltersConv1, filterLenConv1=filterLenConv1, maxPool1=maxPool1,
                                isCBAM1=isCBAM1, reduction_ratio1=reduction_ratio1, 
                                isLayer2=isLayer2, numFiltersConv2=numFiltersConv2, filterLenConv2=filterLenConv2, dilRate2=dilRate2, maxPool2=maxPool2,
                                isCBAM2=isCBAM2, reduction_ratio2=reduction_ratio2, 
                                isLayer3=isLayer3, numFiltersConv3=numFiltersConv3, filterLenConv3=filterLenConv3, dilRate3=dilRate3, maxPool3=maxPool3,
                                isCBAM3=isCBAM3, reduction_ratio3=reduction_ratio3, 
                                is_spatial_transform=is_spatial_transform
                                )
        # if self.isCNN and not use_attention:
        #     self.Embedding = CharCNN(pooltype=pooltype, globalpoolsize=globalpoolsize,
        #                             numFiltersConv1=numFiltersConv1, filterLenConv1=filterLenConv1, maxPool1=maxPool1,
        #                             isLayer2=isLayer2, numFiltersConv2=numFiltersConv2, filterLenConv2=filterLenConv2, dilRate2=dilRate2, maxPool2=maxPool2,
        #                             isLayer3=isLayer3, numFiltersConv3=numFiltersConv3, filterLenConv3=filterLenConv3, dilRate3=dilRate3, maxPool3=maxPool3,)
        
        self.isRNN = isRNN
        if self.isRNN:
            self.RNN = RNN(LSTM_input_size=LSTM_input_size, LSTM_hidden_size=LSTM_hidden_size, LSTM_hidden_layes=LSTM_hidden_layes)
        
        self.isDense = isDense
        if self.isDense:
            self.MLP = MLP(dense_input_size=dense_input_size, dense1=dense1, dropout1=dropout1, 
                            is_fc2=is_fc2, dense2=dense2, dropout2=dropout2, 
                            is_fc3=is_fc3, dense3=dense3, dropout3=dropout3,)
            
        # branches for each task
        for idx, cnt in anno_cnt.items():
            setattr(self, "task_{}".format(idx), MLP(tower_input_size, tower_hidden, tower_drop, 
                                                    is_pred=True, pred_prob=pred_prob, output_size=cnt))

        self.cat_task = ConcatenateTask()

    def forward(self, x):
        logging.debug(x.shape)
        if self.isCNN:
            fmap = self.Embedding(x)
            logging.debug("-----finish convs-----\n")
            logging.debug(fmap.shape)
        else:
            fmap = x
            logging.debug(fmap.shape)

        if self.isRNN:
            fmap = fmap.transpose(-1, 1)
            fmap = self.RNN(fmap)
            fmap = fmap.transpose(-1, 1).mean(-1)
            logging.debug("-----finish LSTM-----\n")
            logging.debug(fmap.shape)
        
        fmap = fmap.view(fmap.size(0), -1) #flatten
        if self.isDense:    
            h = self.MLP(fmap)
            logging.debug("-----finish MLP-----\n")
            logging.debug(h.shape)
        else:
            h = fmap
            logging.debug(h.shape)
            
        outs = []
        for idx, cnt in anno_cnt.items():
            layer = getattr(self, "task_{}".format(idx))
            outs.append(layer(h))
        logging.debug("-----finish Prediction Tower-----\n")

        # out = torch.cat(outs, dim=1)
        out = self.cat_task(outs)
        logging.debug(out.shape)

        return out

class NINCNN(BasicModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 512, 13, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1, 1),
            nn.ReLU(),
            nn.AvgPool1d(13),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 768, 9, 1, dilation=2),
            nn.ReLU(),
            nn.Conv1d(768, 768, 1, 1),
            nn.ReLU(),
            nn.AvgPool1d(4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(768, 1024, 7, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1, 1),
            nn.ReLU(),
        )
        self.GAP = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        logging.debug(x.shape)
        x = self.conv1(x)
        logging.debug(x.shape)
        x = self.conv2(x)
        logging.debug(x.shape)
        x = self.conv3(x)
        logging.debug(x.shape)
        x = self.GAP(x)
        logging.debug(x.shape)
        x = x.view(x.size(0), -1)
        logging.debug(x.shape)
        return x


class NIN(BasicModule):
    def __init__(self, output_size):
        super().__init__()
        self.Embedding = NINCNN()
        self.pred = nn.Sequential(
            nn.Linear(1024, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.Embedding(x)
        x = self.pred(x)
        logging.debug(x.shape)
        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( max_pool )
            # elif pool_type=='lp':
            #     lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            #     channel_att_raw = self.mlp( lp_pool )
            # elif pool_type=='lse':
            #     # LSE pool only
            #     lse_pool = logsumexp_2d(x)
            #     channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).expand_as(x) + 1
        return x * scale # x*(scale+1)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) + 1 # broadcasting
        return x * scale # x*(scale+1)

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def mask3d(value, sizes):
    v_mask = 0
    v_unmask = 1
    mask = value.data.new(value.size()).fill_(v_unmask)
    n = mask.size(1)
    for i, size in enumerate(sizes):
        if size < n:
            mask[i,size:,:] = v_mask
    return Variable(mask) * value


def fill_context_mask(mask, sizes, v_mask, v_unmask):
    mask.fill_(v_unmask)
    n_context = mask.size(2)
    for i, size in enumerate(sizes):
        if size < n_context:
            mask[i,:,size:] = v_mask
    return mask


def dot(a, b):
    return a.bmm(b.transpose(1, 2))


def attend(query, context, value=None, score='dot', normalize='softmax',
           context_sizes=None, context_mask=None, return_weight=False
           ):
    """Attend to value (or context) by scoring each query and context.
    Args
    ----
    query: Variable of size (B, M, D1)
        Batch of M query vectors.
    context: Variable of size (B, N, D2)
        Batch of N context vectors.
    value: Variable of size (B, N, P), default=None
        If given, the output vectors will be weighted
        combinations of the value vectors.
        Otherwise, the context vectors will be used.
    score: str or callable, default='dot'
        If score == 'dot', scores are computed
        as the dot product between context and
        query vectors. This Requires D1 == D2.
        Otherwise, score should be a callable:
             query    context     score
            (B,M,D1) (B,N,D2) -> (B,M,N)
    normalize: str, default='softmax'
        One of 'softmax', 'sigmoid', or 'identity'.
        Name of function used to map scores to weights.
    context_mask: Tensor of (B, M, N), default=None
        A Tensor used to mask context. Masked
        and unmasked entries should be filled 
        appropriately for the normalization function.
    context_sizes: list[int], default=None,
        List giving the size of context for each item
        in the batch and used to compute a context_mask.
        If context_mask or context_sizes are not given,
        context is assumed to have fixed size.
    return_weight: bool, default=False
        If True, return the attention weight Tensor.
    Returns
    -------
    output: Variable of size (B, M, P)
        If return_weight is False.
    weight, output: Variable of size (B, M, N), Variable of size (B, M, P)
        If return_weight is True.

    """
    q, c, v = query, context, value
    if v is None:
        v = c

    batch_size_q, n_q, dim_q = q.size()
    batch_size_c, n_c, dim_c = c.size()
    batch_size_v, n_v, dim_v = v.size()

    if not (batch_size_q == batch_size_c == batch_size_v):
        msg = 'batch size mismatch (query: {}, context: {}, value: {})'
        raise ValueError(msg.format(q.size(), c.size(), v.size()))

    batch_size = batch_size_q

    # Compute scores
    if score == 'dot':
        s = dot(q, c)
    elif callable(score):
        s = score(q, c)
    else:
        raise ValueError(f'unknown score function: {score}')

    # Normalize scores and mask contexts
    if normalize == 'softmax':
        if context_mask is not None:
            s = context_mask + s

        elif context_sizes is not None:
            context_mask = s.data.new(batch_size, n_q, n_c)
            context_mask = fill_context_mask(context_mask,
                                             sizes=context_sizes,
                                             v_mask=float('-inf'),
                                             v_unmask=0
                                             )
            s = context_mask + s

        s_flat = s.view(batch_size * n_q, n_c)
        w_flat = softmax(s_flat, dim=1)
        w = w_flat.view(batch_size, n_q, n_c)

    elif normalize == 'sigmoid' or normalize == 'identity':
        w = sigmoid(s) if normalize == 'sigmoid' else s
        if context_mask is not None:
            w = context_mask * w
        elif context_sizes is not None:
            context_mask = s.data.new(batch_size, n_q, n_c)
            context_mask = fill_context_mask(context_mask,
                                             sizes=context_sizes,
                                             v_mask=0,
                                             v_unmask=1
                                             )
            w = context_mask * w

    else:
        raise ValueError(f'unknown normalize function: {normalize}')

    # Combine
    z = w.bmm(v)
    if return_weight:
        return w, z
    return z


class CharCNN(BasicModule):
    def __init__(self, pooltype = 'maxpool', globalpoolsize=128,
                    numFiltersConv1=128, filterLenConv1=3, maxPool1=3,
                    isLayer2=True, numFiltersConv2=256, filterLenConv2=3, dilRate2=1, maxPool2=3,
                    isLayer3=True, numFiltersConv3=512, filterLenConv3=3, dilRate3=1, maxPool3=3):
        
        super(CharCNN, self).__init__()
        self.isLayer2 = isLayer2
        self.isLayer3 = isLayer3
        if pooltype == 'avgpool':
            poollayer = nn.AvgPool1d
        else:
            poollayer = nn.MaxPool1d
            
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, 
                      out_channels=numFiltersConv1, 
                      kernel_size=filterLenConv1, 
#                       padding=int((filterLenConv1 - 1) / 2), 
                     ),
            nn.BatchNorm1d(numFiltersConv1),
            nn.ReLU(),
            poollayer(maxPool1)
            )
        
        if self.isLayer2:
            self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=numFiltersConv1, 
                          out_channels=numFiltersConv2, 
                          kernel_size=filterLenConv2, 
#                           padding=int((filterLenConv2 - 1) / 2), 
                          dilation=dilRate2
                         ),
                nn.BatchNorm1d(numFiltersConv2),
                nn.ReLU(),
                poollayer(maxPool2)
            )
            
        if self.isLayer2 and self.isLayer3:
            self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=numFiltersConv2, 
                          out_channels=numFiltersConv3, 
                          kernel_size=filterLenConv3, 
#                           padding=int((filterLenConv3 - 1) / 2), 
                          dilation=dilRate3
                         ),
                nn.BatchNorm1d(numFiltersConv3),
                nn.ReLU(),
                poollayer(maxPool3)
            )
        
        self.globalpooling = nn.AdaptiveAvgPool1d(globalpoolsize)
        
    def forward(self, x):
        logging.debug(x.shape)

        out = self.conv1(x)
        logging.debug(out.shape)

        if self.isLayer2:
            out = self.conv2(out)
            logging.debug(out.shape)

        if self.isLayer2 and self.isLayer3:
            out = self.conv3(out)
            logging.debug(out.shape)
        
        out = self.globalpooling(out) # batchsize * filtersize * globalpoolsize
        logging.debug(out.shape)
        
        return out


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
            output = self.model(data)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
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
                output = self.model(inputs)
                test_loss = self.criterion(output, targets)
                batch_losses.append(test_loss.cpu().item())
                all_predictions.append(output.cpu().data.numpy())
                all_targets.append(targets.cpu().data.numpy())
        average_loss = np.average(batch_losses)
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        return average_loss, all_predictions, all_targets

    def get_current_model(self):
        return self.model


def show_train_log(loss_train=None, loss_val=None, acc_val=None, 
                   loss_test=None, acc_test=None,
                   lrs=None,
                    fig_size=(12,8),
                    save=True,
                    output_dir='Figures',
                    output_fname="Training_loss_log.pdf",
                    style="seaborn-colorblind",
                    fig_title="Training Log",
                    dpi=500):
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    plt.style.use(style)
    plt.figure()
    if loss_train:
        plt.plot(range(1, len(loss_train)+1), loss_train, 'b', label='Training Loss')
    if loss_val:
        plt.plot(range(1, len(loss_val)+1), loss_val, 'r', label='Validation Loss')
    if loss_test:
        plt.plot(range(1, len(loss_test)+1), loss_test, 'black', label='Test Loss')
    rate = 1
    if acc_val:
        plt.plot(range(1, len(acc_val)+1), list(map(lambda x: x*rate, acc_val)), 'g', label=str(rate)+'X Validation Accuracy')
    if acc_test:
        plt.plot(range(1, len(acc_test)+1), list(map(lambda x: x*rate, acc_test)), 'purple', label=str(rate)+'X Test Accuracy')
    if lrs:
        rate = int(1/lrs[0])
        plt.plot(range(1, len(lrs)+1), list(map(lambda x: x*rate, lrs)), 'y', label=str(rate)+'X Learning Rates')
    plt.title(fig_title)
    plt.legend()
    if save:
        plt.savefig(os.path.join(output_dir, output_fname),
                    format="pdf", dpi=dpi)
    else:
        plt.show()
    plt.close()


def calculate_roc(target, prediction):
    # assert len(np.shape(prediction))>1, "Input should be y_prediction_Probability"
    if len(np.shape(target)) == 1:
        target = onehot_encode(target)
    fpr, tpr, roc_auc = {}, {}, {} # orderedDict after python3.8
    n_classes = target.shape[-1]
    for index in range(n_classes):
        feature_targets = target[:, index]
        feature_preds = prediction[:, index]
        if len(np.unique(feature_targets)) > 1:
            fpr[index], tpr[index], _ = roc_curve(feature_targets, feature_preds)
            roc_auc[index] = auc(fpr[index], tpr[index])
    fpr['micro'], tpr['micro'], _ = roc_curve(target.ravel(), prediction.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr.get(i, [0]) for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr.get(i, [0]), tpr.get(i, [0]))
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def show_auc_curve(fpr, tpr, roc_auc,
                    fig_size=(10,8),
                    save=True,
                    output_dir='Figures',
                    output_fname='roc_curves.pdf',
                    style="seaborn-colorblind",
                    fig_title="Feature ROC curves",
                    dpi=500):
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    plt.style.use(style)
    plt.figure()
    # n_classes
    n_classes = len(roc_auc) - 2
    # Plot all ROC curves
    plt.figure(figsize=fig_size)
    lw = 1
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    # colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'green', 'grey', 'black', 'yellow', 'purple', 'brown', 'darkblue', 'darkred', 'gold', 'orange', 'pink', 'violet', 'turquoise', 'tomato']
    colors = ["grey"]
    for i, color in zip(range(n_classes), itertools.cycle(colors)):
        plt.plot(fpr.get(i, [0]), tpr.get(i, [0]), color=color, lw=lw,
                #  label='ROC curve of class {0} (area = {1:0.2f})'
                #  ''.format(i, roc_auc.get(i, 0))
                 )
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(fig_title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(os.path.join(output_dir, output_fname),
                    format="pdf", dpi=dpi)
    else:
        plt.show()
    plt.close()


def visualize_roc_curves(target, prediction,
                         fig_size=(10,8),
                         save=True,
                         output_dir='Figures',
                         output_fname='roc_curves.pdf',
                         style="seaborn-colorblind",
                         fig_title="Feature ROC curves",
                         dpi=500):
    """
    Output the ROC curves for each feature predicted by a model
    as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature ROC curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """
#     assert len(np.shape(prediction))>1, "Input should be y_prediction_Probability"
    if len(np.shape(target)) == 1:
        target = onehot_encode(target)
    os.makedirs(output_dir, exist_ok=True)
    # calculate_roc(target, prediction)
    fpr, tpr, roc_auc = calculate_roc(target, prediction)
    show_auc_curve(fpr, tpr, roc_auc,
                    fig_size=fig_size,
                    save=save,
                    output_dir=output_dir,
                    output_fname=output_fname,
                    style=style,
                    fig_title=fig_title,
                    dpi=dpi)



def calculate_pr(target, prediction):
    assert len(np.shape(prediction))>1, "Input should be y_prediction_Probability"
    if len(np.shape(target)) == 1:
        target = onehot_encode(target)
    # For each class
    precision, recall, average_precision = {}, {}, {}
    n_classes = target.shape[-1]
    for index in range(n_classes):
        feature_targets = target[:, index]
        feature_preds = prediction[:, index]
        if len(np.unique(feature_targets)) > 1:
            precision[index], recall[index], _ = precision_recall_curve(feature_targets, feature_preds)
            average_precision[index] = average_precision_score(feature_targets, feature_preds)
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(target.ravel(), prediction.ravel())
    average_precision["micro"] = average_precision_score(target.ravel(), prediction.ravel(), average="micro")
    return precision, recall, average_precision


def show_pr_curve(precision, recall, average_precision,
                    fig_size=(10,8),
                    save=True,
                    output_dir='Figures',
                    output_fname='pr_curves.pdf',
                    style="seaborn-colorblind",
                    fig_title="Feature PR curves",
                    dpi=500):
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("PDF")
    import matplotlib.pyplot as plt
    plt.style.use(style)
    plt.figure()
    # n_classes
    n_classes = len(precision) - 1
    # Plot all ROC curves
    plt.figure(figsize=fig_size)
    lw = 2 

    # setup plot details
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'green', 'grey', 'black', 'yellow', 'purple']

    plt.figure(figsize=fig_size)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=lw)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                    ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fig_title)
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    if save:
        plt.savefig(os.path.join(output_dir, output_fname),
                    format="pdf", dpi=dpi)
    else:
        plt.show()
    plt.close()


def visualize_pr_curves(target, prediction,
                        fig_size=(10,8),
                        save=True,
                        output_dir='Figures',
                        output_fname='pr_curves.pdf',
                        style="seaborn-colorblind",
                        fig_title="Feature precision-recall curves",
                        dpi=500):
    """
    Output the precision-recall (PR) curves for each feature predicted by
    a model as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    report_gt_feature_n_positives : int, optional
        Default is 50. Do not visualize an PR curve for a feature with
        less than 50 positive examples in `target`.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature precision-recall curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """
    os.makedirs(output_dir, exist_ok=True)
    assert len(np.shape(prediction))>1, "Input should be y_prediction_Probability"
    if len(np.shape(target)) == 1:
        target = onehot_encode(target)
    os.makedirs(output_dir, exist_ok=True)
    # calculate_roc(target, prediction)
    precision, recall, average_precision = calculate_pr(target, prediction)
    show_pr_curve(precision, recall, average_precision,
                    fig_size=fig_size,
                    save=save,
                    output_dir=output_dir,
                    output_fname=output_fname,
                    style=style,
                    fig_title=fig_title,
                    dpi=dpi)


def calculate_correlation(target, prediction, method="pearson"):
    if method == "pearson":
        correlation_fn = pearsonr
    elif method == "spearman":
        correlation_fn = spearmanr
    elif method == "kendall":
        correlation_fn = kendalltau
    # assert len(np.shape(prediction))>1, "Input should be y_prediction_Probability"
    if len(np.shape(target)) == 1:
        target = onehot_encode(target)
    correlation, pvalue = {}, {} # orderedDict after python3.8
    n_classes = target.shape[-1]
    for index in range(n_classes):
        feature_targets = target[:, index]
        feature_preds = prediction[:, index]
        if len(np.unique(feature_targets)) > 1:
            correlation[index], pvalue[index] = correlation_fn(feature_targets, feature_preds)
    return correlation, pvalue

LAYER_KEY = 'layers'
NAME_KEY = 'name'
ANCHOR_KEY = 'anchor_layer'
LOSS_KEY = 'loss'
LOSS_REG_KEY = 'loss_weight'
AUTO_WEIGHT_KEY = 'auto'
WEIGHT_INIT_KEY = 'loss_init_val'

MISSING_WEIGHT_MSG = "Expect {0} for task {1} but none provided."

class MTLModel(nn.Module):
    """
    A torch.nn.Module built from a set of shared and task specific layers
    Attributes
    ----------
    g : networkx.Graph
        The meta-computation graph
    task_layers : list
        A list which holds the layers for which to build the computation graph
    output_tasks : list
        A list which holds the tasks for which the output should be returned
    layer_names : list
        A list of the names of each layer
    losses : dict
        A dictionary which maps the name of a layer to its loss function
    loss_weights : dict
        A dictionary which maps the name of a layer to the weight of its loss
        function
    """

    def __init__(self, task_layers, output_tasks):
        super(MTLModel, self).__init__()
        self.task_layers = task_layers
        self.output_tasks = output_tasks
        self.layer_names = [t[NAME_KEY] for t in task_layers]

        self._initialize_graph()

        self._initialize_losses()
        self._initialize_loss_weights()

    def _initialize_losses(self):
        self.losses = {task[NAME_KEY]: task[LOSS_KEY]\
                       for task in self.task_layers if LOSS_KEY in task.keys()}

    def _initialize_loss_weights(self):
        self.loss_weights = {}
        for task in self.task_layers:
            self._set_loss_weight(task)

    def _set_loss_weight(self, task):
        task_name = task[NAME_KEY]
        if LOSS_REG_KEY in task.keys():
            if task[LOSS_REG_KEY] == AUTO_WEIGHT_KEY:
                assert WEIGHT_INIT_KEY in task.keys(),\
                        MISSING_WEIGHT_MSG.format(WEIGHT_INIT_KEY, task_name)
                loss_weight = task[WEIGHT_INIT_KEY]
                loss_name = f'{task_name}_loss'
                loss_weight = torch.nn.Parameter(torch.full((1,),
                                                            loss_weight))
                setattr(self, loss_name, loss_weight)
                self.loss_weights[task_name] = getattr(self, loss_name)
            else:
                self.loss_weights[task_name] = task[LOSS_REG_KEY]
    
    def _initialize_graph(self):
        self.g = nx.DiGraph()
        self.g.add_node('root')
        self._build_graph()

    def _bfs_forward(self, start_node):
        ''' Here we iteratore through the graph in a BFS-fashion starting from
        `start_node`, typically this is the `root` node. This node is skipped
        and we pass the input data and resulting outputs from all layers foward.
        '''
        visited = {node: False for node in self.layer_names}

        # First node is visited
        queue = [start_node]
        visited[start_node] = True

        while queue:
            node = queue.pop(0)
            if node != start_node:
                input_nodes = self.g.predecessors(node)
                if logging.getLogger().level == logging.DEBUG:
                    l = copy(input_nodes)
                    print(f"Feeding output from {list(l)} into {node}")
                cur_layer = getattr(self, node)

                # Get the output from the layers that serve as input
                output_pre_layers = []
                output_complete = True
                for n in input_nodes:
                    # If an output is not ready yet, because that node has not
                    # been computed, we put the current node back into the queue
                    if n not in self.outputs.keys():
                        if logging.getLogger().level == logging.DEBUG:
                            print(f"No output for layer {n} yet")
                        output_complete = False
                        break
                    else:
                        output_pre_layers.append(self.outputs[n])

                if not output_complete:
                    if logging.getLogger().level == logging.DEBUG:
                        print(f"Putting {node} back into the queue.")
                    queue.append(node)
                else:
                    cur_output = cur_layer(*output_pre_layers)
                    self.outputs[node] = cur_output

            for i in self.g.successors(node):
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

        losses, loss_weights = self._get_losses()
        return [self.outputs[t] for t in self.output_tasks], losses, loss_weights

    def _get_losses(self):
        losses = []
        loss_weights = []
        for t in self.output_tasks:
            losses.append(self.losses.get(t))
            loss_weights.append(self.loss_weights.get(t))
        return losses, loss_weights

    def _build_graph(self):
        for layer in self.task_layers:
            self._add_layer(layer)
            self._add_to_graph(layer)

    def _add_to_graph(self, layer):
        layer_name = layer[NAME_KEY]
        self._add_node(layer_name)

        if 'anchor_layer' not in layer.keys():
            # If there is no anchor layer, we expect it to be a layer which
            # receives data inputs and is hence connected to the root node
            self.g.add_edge('root', layer_name)
        else:
            anchor_layer = layer[ANCHOR_KEY]
            if isinstance(anchor_layer, list):
                for a_l_name in anchor_layer:
                    self._add_node(a_l_name)
                    self.g.add_edge(a_l_name, layer_name)
            else:
                self._add_node(anchor_layer)
                self.g.add_edge(anchor_layer, layer_name)

    def _add_node(self, layer):
        if isinstance(layer, str):
            layer_name = layer
            self.g.add_node(layer_name)
        else:
            layer_name = layer[NAME_KEY]
            self.g.add_node(layer_name)
            if 'anchor_layer' not in layer.keys():
                self.g.add_edge('root', layer_name)
    
    def _add_layer(self, layer):
        layer_modules = layer[LAYER_KEY]
        layer_name_main = layer[NAME_KEY]
        setattr(self, layer_name_main, layer_modules)

    def forward(self, input):
        self.outputs = {'root': input}
        return self._bfs_forward('root')