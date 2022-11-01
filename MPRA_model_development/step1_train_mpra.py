import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as tfcallbacks 
from sklearn.model_selection import train_test_split



os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development/archs')

# from arch_tnnbasic import setupTNN
from arch_tnnqtl import setupTNN
from arch_tnnmod import setupTNNmod
from load_data import load_data

def train_mpra(cbpdir, datadir, dataid, mpramodelid, versionid, lr):
    print("here")
    TNN = setupTNNmod('models/' + cbpdir + '/chrombpnet_wo_bias.h5', lr) #MODIFY
    print(TNN.summary())
    XR_train = np.load(datadir + '/XR_train' + dataid + '.npy')
    XA_train = np.load(datadir + '/XA_train' + dataid + '.npy')
    y_train = np.load(datadir + '/delta_train' + dataid + '.npy')
    
    print(XR_train.shape, XA_train.shape, y_train.shape)

    checkpointer = tfcallbacks.ModelCheckpoint(filepath='MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + versionid, monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=5, verbose=1, min_delta=0.0005, restore_best_weights=True)
    cur_callbacks=[checkpointer, earlystopper]
    print("LR:", lr)
    TNN.fit([XR_train, XA_train], y_train, batch_size=16, epochs=40, validation_split=0.2, callbacks=cur_callbacks)
    TNN.save('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + versionid + '.h5')
