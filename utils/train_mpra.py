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

from arch_tnnplus import setupTNN_P
from arch_tnnmod import setupTNN_M
from load_data import load_data

def train_mpra():
    lr = 0.00001
    TNN = setupTNN_M('models/chrombpnet.h5', lr)
    print(TNN.summary())
    XR_train = np.load('data/training/XR_train.npy')
    XR_test = np.load('data/training/XR_test.npy')
    XA_train = np.load('data/training/XA_train.npy')
    XA_test = np.load('data/training/XA_test.npy')
    y_train = np.load('data/training/y_train.npy')
    y_test = np.load('data/training/y_test.npy')
    print(XR_train.shape, XR_test.shape, XA_train.shape, XA_test.shape, y_train.shape, y_test.shape)
    checkpointer = tfcallbacks.ModelCheckpoint(filepath="MPRAModelv4-220904", monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=5, verbose=1, min_delta=0.0005, restore_best_weights=True)
    cur_callbacks=[checkpointer, earlystopper]
    print("LR:", lr)
    TNN.fit([XR_train, XA_train], y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=cur_callbacks)
    TNN.save('MPRAModelv4-220904.h5')

if ('__main__'):
    train_mpra()