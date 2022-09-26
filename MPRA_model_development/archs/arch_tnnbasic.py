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
    return ChromBPNet

def setupTNN(chrombpnetfile, lr):
    AlleleR = Input(shape=(2114, 4))
    AlleleA = Input(shape=(2114, 4))
    print("here")
    
    ChromBPNet = loadCBP(chrombpnetfile)

    EncodedR = ChromBPNet(inputs=[AlleleR])
    EncodedA = ChromBPNet(inputs=[AlleleA])
    print("here")
    MergedR = Lambda(merge)(EncodedR)
    MergedA = Lambda(merge)(EncodedA)

    L1_layer = Lambda(lambda tensors:keras.backend.log(tf.math.divide(tensors[0],tensors[1])))
    L1_distance = L1_layer([MergedR, MergedA])
    print("here")
    X = Dense(64, activation='relu') (L1_distance)
    prediction = Dense(1, activation='tanh')(X)
    TNN = keras.Model(inputs=[AlleleR, AlleleA], outputs=[prediction])
    TNN.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    return TNN