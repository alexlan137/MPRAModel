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

from arch_tnnbasic import setupTNN
from load_data import load_data

def train_mpra():
    lr = 0.0002
    print("here")
    TNN = setupTNN('models/GM12878/chrombpnet_wo_bias.h5', lr)
    print(TNN.summary())
    XR_train = np.load('data/MPRA_partitioned/Kampman/XRKampman.mAL.t1000.p0.3.c300.npy')
    XA_train = np.load('data/MPRA_partitioned/Kampman/XAKampman.mAL.t1000.p0.3.c300.npy')
    y_train = np.load('data/MPRA_partitioned/Kampman/deltaKampman.mAL.t1000.p0.3.c300.npy')
    print(XR_train.shape, XA_train.shape, y_train.shape)
    checkpointer = tfcallbacks.ModelCheckpoint(filepath="MPRA_model_development/models/MPRAModel.Kampman.mAL.t1000.p0.3.c300.220925.v5", monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=5, verbose=1, min_delta=0.0005, restore_best_weights=True)
    cur_callbacks=[checkpointer, earlystopper]
    print("LR:", lr)
    TNN.fit([XR_train, XA_train], y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=cur_callbacks)
    TNN.save("MPRA_model_development/models/MPRAModel.Kampman.mAL.t1000.p0.3.c300.220925.v5")

if ('__main__'):
    train_mpra()