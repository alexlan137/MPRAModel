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

def setupTNN_M(chrombpnetfile, lr):
    inpSeq = Input(shape = (2114, 4))
    ChromBPNet = loadCBP(chrombpnetfile)
    X = ChromBPNet([inpSeq])
    X = Lambda(merge)(X)
    CORE = keras.Model(inputs=[inpSeq], outputs=[X])
    CORE.compile(optimizer=keras.optimizers.Adam())

    AlleleR = Input(shape=(2114, 4))
    AlleleA = Input(shape=(2114, 4))

    EncodedR = CORE(inputs=[AlleleR])
    EncodedA = CORE(inputs=[AlleleA])

    LFC = Lambda(lambda tensors:keras.backend.log(tensors[0]/tensors[1]))
    LFCoutput = LFC([EncodedR, EncodedA])

    dense1 = Dense(256, activation='relu')(LFCoutput)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(16, activation='relu')(dense2)
    prediction = Dense(1, activation='sigmoid')(dense3)
    TNN = keras.Model(inputs=[AlleleR, AlleleA], outputs=[prediction])
    TNN.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return TNN