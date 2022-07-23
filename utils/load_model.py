import tensorflow as tf
from tensorflow import keras

from utils.loss import multinomial_nll

def load(filepath):
    with keras.utils.CustomObjectScope({'multinomial_nll':multinomial_nll, 'tf':tf}):
        model_chrombpnet = keras.models.load_model(filepath)
    print(model_chrombpnet.summary())
    return model_chrombpnet