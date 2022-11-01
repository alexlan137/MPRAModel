import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')

from load_model import load
from merge import merge

def loadCBP(filepath):
    ChromBPNet = load(filepath)
    return ChromBPNet

def root_mean_squared_error(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

def setupTNN(chrombpnetfile, lr):
    AlleleR = Input(shape=(2114, 4))
    AlleleA = Input(shape=(2114, 4))
    
    ChromBPNet = loadCBP(chrombpnetfile)

    EncodedR = ChromBPNet(inputs=[AlleleR])
    EncodedA = ChromBPNet(inputs=[AlleleA])

    MergedR = Lambda(merge)(EncodedR)
    MergedA = Lambda(merge)(EncodedA)

    L1_layer = Lambda(lambda tensors:keras.backend.log(tf.math.divide(tensors[0],tensors[1])))
    L1_distance = L1_layer([MergedR, MergedA])

    X = tf.expand_dims(L1_distance, axis=-1)
    X = Conv1D(filters=16, kernel_size=10, padding='valid', activation='relu')(X)
    X = Flatten()(X)
    X = Dense(48, activation='relu') (X)
    prediction = Dense(1, activation='tanh')(X)
    TNN = keras.Model(inputs=[AlleleR, AlleleA], outputs=[prediction])
    TNN.compile(optimizer=Adam(learning_rate=lr), loss=root_mean_squared_error, metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return TNN