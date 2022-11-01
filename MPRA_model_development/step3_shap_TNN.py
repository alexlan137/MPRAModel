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
from load_data import load_data
from shaputils import combine_mult_and_diffref

from matplotlib import pyplot as plt

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

def flatten(l):
    return [item for sublist in l for item in sublist]

def shap_TNN(mpramodelid, version, datadir, dataid):
    tf.compat.v1.disable_v2_behavior()
    TNN = tf.keras.models.load_model('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version, custom_objects={'root_mean_squared_error':root_mean_squared_error})
    print("TNN loaded from file")
    print(TNN.summary())
    print(TNN.inputs)
    print(TNN.outputs)

    XR = np.load(datadir + '/XR' + dataid + '.npy')
    XA = np.load(datadir + '/XA' + dataid + '.npy')
    y = np.load(datadir + '/delta' + dataid + '.npy')

    XR_train = np.load(datadir + '/XR_train' + dataid + '.npy')
    XA_train = np.load(datadir + '/XA_train' + dataid + '.npy')
    y_train = np.load(datadir + '/delta_train' + dataid + '.npy')

    XR_test = np.load(datadir + '/XR_test' + dataid + '.npy')
    XA_test = np.load(datadir + '/XA_test' + dataid + '.npy')
    y_test = np.load(datadir + '/delta_test' + dataid + '.npy')

    counts_model_input = TNN.input
    print(counts_model_input)
    seq1 = np.expand_dims(XR_test[0], 0)
    seq2 = np.expand_dims(XA_test[0], 0)
    counts_input = [seq1, seq2]
    print(counts_input)

    #generates shuffled sequences - score of difference from reference
    profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
        (counts_model_input, tf.reduce_sum(TNN.outputs[0], axis=-1)),
        shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)
    
    counts_shap_scores = profile_model_counts_explainer.shap_values(
        counts_input, progress_message=10)
    
    print(counts_shap_scores)

    center = 1056
    diff = 150 # CHANGED DIFF (was 40)
    start, end = center - diff, center + diff + 1
    imp1 = get_imp(counts_shap_scores[0], counts_input[0], start, end)
    imp2 = get_imp(counts_shap_scores[1], counts_input[1], start, end)
    delta_scores = imp1-imp2

    print(imp1, imp2, delta_scores)

    minval, maxval = get_range_chrombpnet(imp1, imp2)
    mindelta, maxdelta = get_minmax_chrombpnet(delta_scores)

    title1 = "imp1"
    title2 = "imp2"
    title3 = "imp1-imp2"
    altshap = viz_sequence.plot_weights(array=imp1, title=title1, filepath='shap/altimp.png', minval=minval, maxval=maxval, color="lightsteelblue", figsize=(30, 4))
    refshap = viz_sequence.plot_weights(array=imp2, title=title2, filepath='shap/refimp.png', minval=minval, maxval=maxval, color="lightsteelblue", figsize=(30, 4))
    delshap = viz_sequence.plot_weights(array=delta_scores, title=title3, filepath='shap/delta.png', minval=mindelta, maxval=maxdelta, color="lightsteelblue", figsize=(30, 4))

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
