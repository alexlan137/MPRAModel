import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tensorflow import keras


def merge(CBPout):
    logits = CBPout[0]
    logcts = CBPout[1]

    profile = keras.layers.Softmax(name='softmax')(logits)
    counts = keras.layers.ELU(name='counts')(logcts)
    return profile * counts
