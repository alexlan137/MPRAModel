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

def eval_mpra():
    TNN = tf.keras.models.load_model('MPRA_model_development/models/MPRAModel.Kampman.mAL.t1000.p0.3.c300.220925.v5')
    print("TNN loaded from file")
    print(TNN.summary())

    XR_train = np.load('data/MPRA_partitioned/Kampman/XRKampman.mAL.t1000.p0.3.c300.npy')
    XA_train = np.load('data/MPRA_partitioned/Kampman/XAKampman.mAL.t1000.p0.3.c300.npy')
    y_train = np.load('data/MPRA_partitioned/Kampman/deltaKampman.mAL.t1000.p0.3.c300.npy')

    TNNpreds = TNN.predict([XR_train, XA_train], batch_size=16, verbose=True)
    np.save('MPRA_model_development/models/MPRAModel.Kampman.mAL.t1000.p0.3.c300.220925.v4/preds.v5', np.array(TNNpreds))
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

if ('__main__'):
    eval_mpra()