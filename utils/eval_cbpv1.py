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
import scipy

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')

from merge import merge
from load_model import load
from arch_tnnplus import *


def eval_mpra():
    CBP = load('models/Cluster1.h5')
    print("CBP loaded from file")
    print(CBP.summary())

    XR_test = np.load('data/training/XR_test.npy')
    XA_test = np.load('data/training/XA_test.npy')
    y_test = np.load('data/training/y_test.npy')

    CBPpredsR = CBP.predict([XR_test, np.zeros((len(XR_test), 1000)), np.zeros(len(XR_test))], verbose=True)[:][1] #1
    CBPpredsA = CBP.predict([XA_test, np.zeros((len(XA_test), 1000)), np.zeros(len(XA_test))], verbose=True)[:][1] #1
    CBPpreds = []
    for i in range(len(CBPpredsR)):
        CBPpreds.append(CBPpredsR[i] - CBPpredsA[i])
        # CBPpreds.append(scipy.spatial.distance.jensenshannon(CBPpredsR[i], CBPpredsA[i]))

    results = pd.DataFrame()
    results.insert(0, 'CBP', CBPpreds)
    results.insert(1, 'Y', y_test)
    results = results.sort_values(by=['Y'], ascending=False)
    results.to_csv('metrics/CBPv1v2testing/cbpv1metrics.csv')
    np.save('metrics/CBPv1v2testing/cbpv1preds', np.array(CBPpreds))
    print(results.info())
    print(results.head(20))

    spearmanCBP = spearmanr(y_test, CBPpreds)[0]
    pearsonCBP = pearsonr(y_test, CBPpreds)[0]

    print("spearman CBP:", spearmanCBP)
    print("pearson CBP:", pearsonCBP)


if ('__main__'):
    eval_mpra()