import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from scipy.stats import spearmanr, pearsonr

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development/archs')

from arch_tnnbasic import setupTNN
from merge import merge
from load_model import load
from load_data import load_data

from matplotlib import pyplot as plt

def flatten(l):
    return [item for sublist in l for item in sublist]

def eval_mpra(mpramodelid, datadir, dataid, version):
    TNN = tf.keras.models.load_model('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version)
    print("TNN loaded from file")
    print(TNN.summary())

    XR_train = np.load(datadir + '/XR_train' + dataid + '.npy')
    XA_train = np.load(datadir + '/XA_train' + dataid + '.npy')
    y_train = np.load(datadir + '/delta_train' + dataid + '.npy')

    XR_test = np.load(datadir + '/XR_test' + dataid + '.npy')
    XA_test = np.load(datadir + '/XA_test' + dataid + '.npy')
    y_test = np.load(datadir + '/delta_test' + dataid + '.npy')

    TNNpreds_train = TNN.predict([XR_train, XA_train], batch_size=16, verbose=True)
    np.save('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/preds.train', np.array(TNNpreds_train))
    TNNpreds_train = flatten(TNNpreds_train)
    print(TNNpreds_train)
    print(y_train)
    results_train = pd.DataFrame()
    results_train['TNN'] = TNNpreds_train
    results_train['Y'] = y_train
    results_train.sort_values(by=['Y'])
    print(results_train.info())
    print(results_train.head(20))

    TNNpreds_test = TNN.predict([XR_test, XA_test], batch_size=16, verbose=True)
    np.save('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/preds.test', np.array(TNNpreds_test))
    TNNpreds_test = flatten(TNNpreds_test)
    print(TNNpreds_test)
    print(y_test)
    results_test = pd.DataFrame()
    results_test['TNN'] = TNNpreds_test
    results_test['Y'] = y_test
    results_test.sort_values(by=['Y'])
    print(results_test.info())
    print(results_test.head(20))

    spearmanTNN_test = spearmanr(y_test, TNNpreds_test)[0]
    pearsonTNN_test = pearsonr(y_test, TNNpreds_test)[0]
    
    print("spearman TNN:", spearmanTNN_test)
    print("pearson TNN:", pearsonTNN_test)

    fig, ax = plt.subplots(figsize = (10, 10))
    _, bins, _ = ax.hist(y_train, bins=20, color='coral', alpha=0.6, label = 'mpra')
    ax.hist(TNNpreds_train, bins=bins, color='darkblue', alpha=0.6, label = 'preds_train')
    ax.hist(TNNpreds_test, bins=bins, color='black', alpha=0.6, label = 'preds_test')
    plt.legend()
    plt.savefig('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/distribution.all.png')

    fig, ax = plt.subplots(figsize = (10, 10))
    results_train.plot.scatter(x='TNN', y='Y', c='darkblue', alpha=0.6, ax=ax)
    results_test.plot.scatter(x='TNN', y='Y', c='coral', alpha=0.6, ax=ax)
    plt.savefig('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/correlation.all.png')

    with open('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/metrics.txt', 'w') as f:
        f.write('spearman TNN: ')
        f.write(str(spearmanTNN_test))
        f.write('\n')
        f.write('pearson TNN: ')
        f.write(str(pearsonTNN_test))
        f.write('\n')
