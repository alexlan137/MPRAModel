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

def flatten(l):
    return [item for sublist in l for item in sublist]

def eval_mpra(mpramodelid, datadir, dataid, version):
    TNN = tf.keras.models.load_model('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version)
    print("TNN loaded from file")
    print(TNN.summary())

    XR_train = np.load(datadir + '/XR_test' + dataid + '.npy')
    XA_train = np.load(datadir + '/XA_test' + dataid + '.npy')
    y_train = np.load(datadir + '/delta_test' + dataid + '.npy')

    TNNpreds = TNN.predict([XR_train, XA_train], batch_size=16, verbose=True)
    np.save('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/preds', np.array(TNNpreds))
    TNNpreds = flatten(TNNpreds)
    print(TNNpreds)
    print(y_train)
    results = pd.DataFrame()
    results['TNN'] = TNNpreds
    results['Y'] = y_train
    results.sort_values(by=['Y'])
    print(results.info())
    print(results.head(20))

    spearmanTNN = spearmanr(y_train, TNNpreds)[0]
    pearsonTNN = pearsonr(y_train, TNNpreds)[0]
    
    print("spearman TNN:", spearmanTNN)
    print("pearson TNN:", pearsonTNN)

    with open('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/metrics.txt', 'w') as f:
        f.write('spearman TNN: ')
        f.write(str(spearmanTNN))
        f.write('\n')
        f.write('pearson TNN: ')
        f.write(str(pearsonTNN))
        f.write('\n')
