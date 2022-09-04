import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')

from load_model import load
from merge import merge

def loadCBP(filepath):
    ChromBPNet = load(filepath)
    ChromBPNet.trainable = False
    return ChromBPNet

def setupTNN(chrombpnetfile, lr):
    inpSeq = Input(shape = (2114, 4))
    ChromBPNet = loadCBP(chrombpnetfile)
    X = ChromBPNet([inpSeq], training=False)
    X = Lambda(merge)(X)
    CORE = keras.Model(inputs=[inpSeq], outputs=[X])
    CORE.compile(optimizer=keras.optimizers.Adam())
    print("———CORE MODEL ARCHITECTURE———")
    print(CORE.summary())

    AlleleR = Input(shape=(2114, 4))
    AlleleA = Input(shape=(2114, 4))

    EncodedR = CORE(inputs=[AlleleR], training=False)
    EncodedA = CORE(inputs=[AlleleA], training=False)

    L1_layer = Lambda(lambda tensors:keras.backend.abs(tensors[0]-tensors[1]))
    L1_distance = L1_layer([EncodedR, EncodedA])
    L1_distance = tf.expand_dims(L1_distance, axis=-1)

    # merged = Conv1D(filters=64, kernel_size=20, activation='sigmoid')(L1_distance)
    dense1 = Dense(512, activation='relu')(L1_distance)
    dense2 = Dense(128, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    prediction = Dense(1, activation='sigmoid')(dense3)
    TNN = keras.Model(inputs=[AlleleR, AlleleA], outputs=[prediction])
    TNN.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return TNN

if __name__ == '__main__':
    TNN = setupTNN('models/chrombpnet.h5', 0.01)
    print(TNN.summary())