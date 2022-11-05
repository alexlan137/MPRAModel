import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import pysam
from mseqgen.sequtils import one_hot_encode
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense, Flatten, Cropping1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.callbacks as tfcallbacks 
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt


os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development')


SNPs = pd.read_csv('data/GWAS/Alzheimers_Bellenguez_2022.all_ld.bed', sep='\t')
print(SNPs.info())
print(SNPs.head())

data_dir = 'data/GWAS/Bellenguez/'
XA = np.load(data_dir + 'XA.npy')
XB = np.load(data_dir + 'XB.npy')

TNN = tf.keras.models.load_model('MPRA_model_development/models/STSCONVS4')
print(TNN.summary())

TNNpreds = TNN.predict([XA, XB], batch_size=16, verbose=True)
SNPs['TNN'] = TNNpreds
SNPs.to_csv('data/GWAS/Bellenguez/scored_TNNV1.csv', seq='\t')