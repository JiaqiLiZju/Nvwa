import sys, os, h5py, pickle, logging, time
import pandas as pd
from optparse import OptionParser
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from keras.metrics import *
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=time.strftime('./log_xpresso.' + '%m%d.%H:%M:%S.txt'),
                    filemode='w')

global X_trainhalflife, X_trainpromoter, y_train, geneName_train, X_validhalflife, X_validpromoter, y_valid, geneName_valid, X_testhalflife, X_testpromoter, y_test, geneName_test, params

data = "/media/ggj/Files/NvWA/nvwa-imputation-official/Human/Dataset.Human_Chrom8_train_test.h5"
mode = "Train"

# unpack datasets
h5file = h5py.File(data, 'r')
celltype = h5file["celltype"][:]
if mode != "test":
    X_trainpromoter = h5file["train_data"][:].astype(np.float32)
    y_train = h5file["train_label"][:].astype(np.float32)
    X_trainhalflife = np.zeros((X_trainpromoter.shape[0], 8), dtype=np.float32)
    geneName_train = h5file["train_gene"][:]

    logging.info(X_trainpromoter.shape)
    logging.info(X_trainhalflife.shape)
    logging.info(y_train.shape)

    X_validpromoter = h5file["val_data"][:].astype(np.float32)
    y_valid = h5file["val_label"][:].astype(np.float32)
    X_validhalflife = np.zeros((X_validpromoter.shape[0], 8), dtype=np.float32)
    geneName_valid = h5file["val_gene"][:]

    logging.info(X_validpromoter.shape)
    logging.info(X_trainhalflife.shape)
    logging.info(y_valid.shape)

X_testpromoter = h5file["test_data"][:].astype(np.float32)
y_test = h5file["test_label"][:].astype(np.float32)
X_testhalflife = np.zeros((X_testpromoter.shape[0], 8), dtype=np.float32)
geneName_test = h5file["test_gene"][:]

logging.info(X_testpromoter.shape)
logging.info(X_testhalflife.shape)
logging.info(y_test.shape)

h5file.close()

X_trainpromoter= X_trainpromoter.swapaxes(1,-1)
X_validpromoter= X_validpromoter.swapaxes(1,-1)
X_testpromoter= X_testpromoter.swapaxes(1,-1)

output_size = y_test.shape[-1]

params = {'datadir': 'pM10Kb_1KTest', 
          'batchsize': 128, 
          'leftpos': 3000, 
          'rightpos': 13500, 
          'activationFxn': 'relu', 
          'numFiltersConv1': 128, 
          'filterLenConv1': 6, 
          'dilRate1': 1, 
          'maxPool1': 30, 
          'numconvlayers': {'numFiltersConv2': 32, 
                            'filterLenConv2': 9, 
                            'dilRate2': 1, 
                            'maxPool2': 10, 
                            'numconvlayers1': {'numconvlayers2': 'two'}
                           }, 
          'dense1': 64, 
          'dropout1': 0.00099, 
          'numdenselayers': {'layers': 'two', 
                             'dense2': 2, 
                             'dropout2': 0.01546}}
          
params['datadir'] = "./data"
params['subsample'] = False
params['cvfold'] = "1"
params['trial'] = "trail"
params['usemodel'] = False
params['tuneMode'] = False #enable mode that trains best model structure over up to 100 epochs, and evaluates final model on test set

leftpos = int(params['leftpos'])
rightpos = int(params['rightpos'])
activationFxn = params['activationFxn']

global X_trainhalflife, y_train
X_trainpromoterSubseq = X_trainpromoter[:,leftpos:rightpos,:]
X_validpromoterSubseq = X_validpromoter[:,leftpos:rightpos,:]
halflifedata = Input(shape=(X_trainhalflife.shape[1:]), name='halflife')
input_promoter = Input(shape=X_trainpromoterSubseq.shape[1:], name='promoter')

mse = 1
if params['usemodel']:
    model = load_model(params['usemodel'])
    print('Loaded results from:', params['usemodel'])
else:
    x = Conv1D(int(params['numFiltersConv1']), int(params['filterLenConv1']), dilation_rate=int(params['dilRate1']), padding='same', kernel_initializer='glorot_normal', input_shape=X_trainpromoterSubseq.shape[1:],activation=activationFxn)(input_promoter)
    x = MaxPooling1D(int(params['maxPool1']))(x)

    if params['numconvlayers']['numconvlayers1'] != 'one':
        maxPool2 = int(params['numconvlayers']['maxPool2'])
        x = Conv1D(int(params['numconvlayers']['numFiltersConv2']), int(params['numconvlayers']['filterLenConv2']), dilation_rate=int(params['numconvlayers']['dilRate2']), padding='same', kernel_initializer='glorot_normal',activation=activationFxn)(x) #[2, 3, 4, 5, 6, 7, 8, 9, 10]
        x = MaxPooling1D(maxPool2)(x)
        if params['numconvlayers']['numconvlayers1']['numconvlayers2'] != 'two':
            maxPool3 = int(params['numconvlayers']['numconvlayers1']['maxPool3'])
            x = Conv1D(int(params['numconvlayers']['numconvlayers1']['numFiltersConv3']), int(params['numconvlayers']['numconvlayers1']['filterLenConv3']), dilation_rate=int(params['numconvlayers']['numconvlayers1']['dilRate3']), padding='same', kernel_initializer='glorot_normal',activation=activationFxn)(x) #[2, 3, 4, 5]
            x = MaxPooling1D(maxPool3)(x)
            if params['numconvlayers']['numconvlayers1']['numconvlayers2']['numconvlayers3'] != 'three':
                maxPool4 = int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['maxPool4'])
                x = Conv1D(int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['numFiltersConv4']), int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['filterLenConv4']), dilation_rate=int(params['numconvlayers']['numconvlayers1']['numconvlayers2']['dilRate4']), padding='same', kernel_initializer='glorot_normal',activation=activationFxn)(x) #[2, 3, 4, 5]
                x = MaxPooling1D(maxPool4)(x)

    x = Flatten()(x)
    x = Concatenate()([x, halflifedata])
    x = Dense(int(params['dense1']))(x)
    x = Activation(activationFxn)(x)
    x = Dropout(params['dropout1'])(x)
    if params['numdenselayers']['layers'] == 'two':
        x = Dense(int(params['numdenselayers']['dense2']))(x)
        x = Activation(activationFxn)(x)
        x = Dropout(params['numdenselayers']['dropout2'])(x)
    main_output = Dense(output_size, activation="sigmoid")(x)
    model = Model(inputs=[input_promoter, halflifedata], outputs=[main_output])
    model.compile(Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), 'binary_crossentropy', metrics=['binary_crossentropy'])

print(model.summary())
plot_model(model, to_file=os.path.join(params['datadir'], 'best_model.png')) #requires Pydot/Graphviz to generate graph of network
X_testpromoterSubseq = X_testpromoter[:,leftpos:rightpos,:]
if not params['usemodel']:
    if params['subsample'] > 0:
        X_trainpromoterSubseq = X_trainpromoterSubseq[0:params['subsample'],:,:]
        X_trainhalflife = X_trainhalflife[0:params['subsample'],:]
        y_train = y_train[0:params['subsample']]
    check_cb = ModelCheckpoint(os.path.join(params['datadir'], params['trial']+params['cvfold']+'trainepoch.{epoch:02d}-{val_loss:.4f}.h5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    result = model.fit([X_trainpromoterSubseq, X_trainhalflife], y_train, batch_size=int(params['batchsize']), shuffle="batch", epochs=100,
        validation_data=[[X_validpromoterSubseq, X_validhalflife], y_valid], callbacks=[earlystop_cb, check_cb])
    mse_history = result.history['val_binary_crossentropy']
    mse = min(mse_history)
    best_file = os.path.join(params['datadir'], params['trial']+params['cvfold']+'trainepoch.%02d-%.4f.h5' % (mse_history.index(mse), mse))
    model = load_model(best_file)
    print('Loaded results from:', best_file)

predictions_test = model.predict([X_testpromoterSubseq, X_testhalflife], batch_size=20).flatten()
slope, intercept, r_value, p_value, std_err = stats.linregress(predictions_test, y_test)
print('Test R^2 = %.3f' % r_value**2)
df = pd.DataFrame(np.column_stack((geneName_test, predictions_test, y_test)), columns=['Gene','Pred','Actual'])
df.to_csv(os.path.join(params['datadir'], params['trial']+params['cvfold']+'predictions.txt'), index=False, header=True, sep='\t')

model = load_model("./data/trail1trainepoch.08-0.4165.h5")
print('Loaded results from:', best_file)

test_pred_prob = model.predict([X_testpromoterSubseq, X_testhalflife], batch_size=20)

from sklearn.metrics import *
from numpy import interp
target, prediction = y_test, test_pred_prob

roc_l = []
n_classes = target.shape[-1]
for index in range(n_classes):
    feature_targets = target[:, index]
    feature_preds = prediction[:, index]
    if len(np.unique(feature_targets)) > 1:
        fpr, tpr, _ = roc_curve(feature_targets, feature_preds)
        roc_l.append(auc(fpr, tpr))

test_r = np.mean(roc_l)
print(test_r)

pd.DataFrame(roc_l, index=celltype, columns=['AUROC_value']).to_csv("./Test/test_mode_roc.csv")
pd.DataFrame(target, columns=celltype, index=geneName_test).to_pickle("./Test/test_target_prob.p", compression='xz')
pd.DataFrame(prediction, columns=celltype, index=geneName_test).to_pickle("./Test/test_mode_pred_prob.p", compression='xz')