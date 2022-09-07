import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from scipy.stats import spearmanr, pearsonr

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')

from merge import merge
from load_model import load
from arch_tnnplus import *
from arch_tnnmod import *

def eval_mpra():
    TNN = tf.keras.models.load_model('MPRAModelv3-220904')
    CBP = load('models/chrombpnet.h5')
    print("TNN loaded from file")
    print(TNN.summary())
    print("CBP loaded from file")
    print(CBP.summary())

    XR_test = np.load('data/training/XR_test.npy')
    XA_test = np.load('data/training/XA_test.npy')
    y_test = np.load('data/training/y_test.npy')

    TNNpreds = TNN.predict([XR_test, XA_test], batch_size=64, verbose=True)
    np.save('metrics/TNNpreds', np.array(TNNpreds))
    CBPpredsR = CBP.predict([XR_test], batch_size=64, verbose=True)[:][1]
    CBPpredsA = CBP.predict([XA_test], batch_size=64, verbose=True)[:][1]
    CBPpreds = []
    for i in range(len(CBPpredsR)):
        CBPpreds.append(CBPpredsR[i] / CBPpredsA[i])

    results = pd.DataFrame()
    results['TNN'] = TNNpreds
    results['CBP'] = CBPpreds
    results['Y'] = y_test
    results.sort_values(by=['Y'])
    print(results.info())
    print(results.head(20))
    
    
    # print (type(TNNpreds), type(CBPpredsR), type(CBPpredsA), type(CBPpreds))
    # print(TNNpreds)
    # print(CBPpredsR)
    # print(CBPpredsA)
    # print(CBPpreds)

    spearmanTNN = spearmanr(y_test, TNNpreds)[0]
    pearsonTNN = pearsonr(y_test, TNNpreds)[0]
    spearmanCBP = spearmanr(y_test, CBPpreds)[0]
    pearsonCBP = pearsonr(y_test, CBPpreds)[0]

    print("spearman TNN:", spearmanTNN)
    print("pearson TNN:", pearsonTNN)
    print("spearman CBP:", spearmanCBP)
    print("pearson CBP:", pearsonCBP)


if ('__main__'):
    eval_mpra()