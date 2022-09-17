import tensorflow as tf
from tensorflow import keras
import sys

sys.path.append('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')


from utils.loss import multinomial_nll

def load(filepath):
    with keras.utils.CustomObjectScope({'multinomial_nll':multinomial_nll, 'tf':tf}):
        model_chrombpnet = keras.models.load_model(filepath)
    print(model_chrombpnet.summary())
    return model_chrombpnet

# if ('__main__'):
#     load('models/chrombpnet.h5')