import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')

from load_model import load
from merge import merge

ChromBPNet = load('models/C1/Cluster1.h5')
ChromBPNet.trainable = False

inpSeq = keras.Input(shape = (2114, 4))
inpBiasL = keras.Input(shape = (1000, ))
inpBiasC = keras.Input(shape = (1, ))
X = ChromBPNet([inpSeq, inpBiasL, inpBiasC], training=False)
X = keras.layers.Lambda(merge)(X)
TNN = keras.Model(inputs=[inpSeq, inpBiasL, inpBiasC], outputs=[X])
TNN.compile(optimizer=keras.optimizers.Adam())
print(TNN.summary())
