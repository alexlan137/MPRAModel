import numpy as np
import pandas as pd
import pysam
import shap
import tensorflow as tf
import sys
import os



os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development/archs')

from modisco.visualization import viz_sequence
from shaputils import *
from arch_tnnbasic import setupTNN
from merge import merge
from load_model import load
from load_data import load_data, load_sequences
from shaputils import combine_mult_and_diffref
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense, Flatten, Cropping1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.callbacks as tfcallbacks 
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr

from matplotlib import pyplot as plt

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

def flatten(l):
    return [item for sublist in l for item in sublist]

def shap_TNN(mpramodelid, datadir, dataid):
    tf.compat.v1.disable_v2_behavior()
    TNN = tf.keras.models.load_model('MPRA_model_development/models/' + mpramodelid)
    print("TNN loaded from file")
    print(TNN.summary())
    print(TNN.inputs)
    print(TNN.outputs)

    # peaks_df = pd.DataFrame(columns = ['chrom', 'pos', 'alt', 'ref'])
    peaks_df = pd.DataFrame()
    chrom = ['chr11']
    pos = [60251677]
    ref = ['T']
    alt = ['C']
    peaks_df['chrom'] = chrom
    peaks_df['pos'] = pos
    peaks_df['ref'] = ref
    peaks_df['alt'] = alt
    XR, XA, seqR, seqA = load_sequences(peaks_df)
    # counts_model_input = TNN.input
    # print(counts_model_input)
    # seq1 = np.expand_dims(XR_test[0], 0)
    # seq2 = np.expand_dims(XA_test[0], 0)
    # counts_input = [seq1, seq2]
    # print(counts_input)

    # #generates shuffled sequences - score of difference from reference
    # profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
    #     (counts_model_input, tf.reduce_sum(TNN.outputs[0], axis=-1)),
    #     shuffle_several_times,
    #     combine_mult_and_diffref=combine_mult_and_diffref)
    # counts_shap_scores = profile_model_counts_explainer.shap_values(
    #     counts_input, progress_message=10)

    counts_model_input = [TNN.input[0], TNN.input[1]]
    counts_input = [XA, XR]
    
    profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
        (counts_model_input, tf.reduce_sum(TNN.outputs[0], axis=-1)),
        shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)
    
    counts_shap_scores = profile_model_counts_explainer.shap_values(
        counts_input, progress_message=10)
    
    print(len(counts_shap_scores))
    print(len(counts_shap_scores[0]))
    print(len(counts_shap_scores[0][0]))
    
    XAshap = counts_shap_scores[0]
    XRshap = counts_shap_scores[1]

    center = 1056
    diff = 50 # CHANGED DIFF (was 40)
    start, end = center - diff, center + diff + 1
    imp1 = get_imp(XRshap[0], XR[0], start, end)
    imp2 = get_imp(XAshap[0], XA[0], start, end)
    delta_scores = imp1-imp2

    # print(imp1, imp2, delta_scores)

    minval, maxval = get_range_chrombpnet(imp1, imp2)
    mindelta, maxdelta = get_minmax_chrombpnet(delta_scores)

    title1 = "imp1"
    title2 = "imp2"
    title3 = "imp1-imp2"
    altshap = viz_sequence.plot_weights(array=imp1, title=title1, filepath='MPRA_model_development/shap/altimpV2.png', minval=minval, maxval=maxval, color="lightsteelblue", figsize=(30, 4))
    refshap = viz_sequence.plot_weights(array=imp2, title=title2, filepath='MPRA_model_development/shap/refimpV2.png', minval=minval, maxval=maxval, color="lightsteelblue", figsize=(30, 4))
    delshap = viz_sequence.plot_weights(array=delta_scores, title=title3, filepath='MPRA_model_development/shap/deltaV2.png', minval=mindelta, maxval=maxdelta, color="lightsteelblue", figsize=(30, 4))

def get_imp(scores, seqs, start, end):
    """ Combines importance scores with the one-hot-encoded sequence to find the
        shap scores for the active bases
    """
    scores = np.asarray(scores)
    seqs = np.asarray(seqs)
    vals = np.multiply(scores, seqs)
    return vals[start:end]

def get_range_chrombpnet(shap1, shap2):
    """ Calculates the y range for the importance score graphs for the individual alleles
        with a buffer of 20% from the min and max values
    """
    minval = min(np.amin(shap1), np.amin(shap2))
    maxval = max(np.amax(shap1), np.amax(shap2))
    buffer = 0.2 * (maxval-minval)
    minval-=buffer
    maxval+=buffer
    return minval, maxval

def get_minmax_chrombpnet(shap1):
    """ Gets the y range for the delta score graph that uses an independent scale
        with a buffer of 20% from the min and max values
    """
    minval = np.amin(shap1)
    maxval = np.amax(shap1)
    buffer = 0.2 * (maxval-minval)
    minval-=buffer
    maxval+=buffer
    return minval, maxval



if('__main__'):
    mpramodelid = 'STS.0KRelu.1.3.16.1.32'
    datadir = 'data/MPRA_partitioned/KampmanRelu'
    dataid = 'KampmanRelu.mSK_K562.t100.p0.8.c300'
    shap_TNN(mpramodelid, datadir, dataid)

    