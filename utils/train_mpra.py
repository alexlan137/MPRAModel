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

from arch_tnnplus import setupTNN
from load_data import load_data

def train_mpra():
    lr = 0.00001
    TNN = setupTNN('models/chrombpnet.h5', lr)
    XR, XA, seqR, seqA, y = load_data('data/MPRA/train-abell-filtered.csv')
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, train_size=0.9)
    print(XR_train.shape, XR_test.shape, XA_train.shape, XA_test.shape, y_train.shape, y_test.shape)
    checkpointer = tfcallbacks.ModelCheckpoint(filepath="MPRAModelv1-220903", monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=5, verbose=1, min_delta=0.001, restore_best_weights=True)
    cur_callbacks=[checkpointer, earlystopper]
    print(TNN.get_config())
    print("LR:", lr)
    TNN.fit([XR, XA], y, batch_size=128, epochs=100, validation_split=0.2, callbacks=cur_callbacks)
    TNN.save('MPRAModelv1-220903.h5')

if ('__main__'):
    train_mpra()