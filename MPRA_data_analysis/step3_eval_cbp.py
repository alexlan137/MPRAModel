import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from scipy.stats import spearmanr, pearsonr
import scipy
from scipy.special import softmax

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')

from merge import merge
from load_model import load

def log_fold_change(ref_track, alt_track, low_bound, up_bound):
    """ Calculates the log fold change track from the alternate and reference predicted profiles
        Returns the predicted lfc track and the graphing range
    """
    # Avoid any divide by 0 errors
    ref_track = ref_track + 0.0001
    alt_track = alt_track + 0.0001
    
    # Generate LFC track with ref / alt
    track = np.divide(ref_track, alt_track)
    track = np.log2(track)

    refscore = np.sum(ref_track[low_bound:up_bound])
    altscore = np.sum(alt_track[low_bound:up_bound])
    lfc = np.log2(refscore / altscore)

    return track, lfc

def eval_cbp(model, crop, id, part_dir, pred_dir):
    # CBP = load('models/chrombpnet_wo_bias.h5')
    CBP = load(model)
    print("CBP loaded from file")
    print(CBP.summary())

    XR_test = np.load(part_dir + 'XR' + id + '.npy')
    XA_test = np.load(part_dir + 'XA' + id + '.npy')
    y_test = np.load(part_dir + 'delta' + id + '.npy')

    CBPpredsR = CBP.predict([XR_test], verbose=True)
    CBPpredsR_logits = CBPpredsR[:][0]
    CBPpredsR_logcts = CBPpredsR[:][1]
    CBPpredsA = CBP.predict([XA_test], verbose=True)
    CBPpredsA_logits = CBPpredsA[:][0]
    CBPpredsA_logcts = CBPpredsA[:][1]
    
    CBPpreds = []
    for i in range(len(CBPpredsR_logits)):
        CBPpredsR_profile = softmax(CBPpredsR_logits[i]) * (np.exp(CBPpredsR_logcts[i]) - 1)
        CBPpredsA_profile = softmax(CBPpredsA_logits[i]) * (np.exp(CBPpredsA_logcts[i]) - 1)
        # print(CBPpredsR_profile)
        # print(CBPpredsA_profile)
        track, lfc = log_fold_change(CBPpredsR_profile, CBPpredsA_profile, int(500 - (crop / 2)), int(500 + (crop / 2)))
        CBPpreds.append(lfc)
        # CBPpreds.append(scipy.spatial.distance.jensenshannon(CBPpredsR[i], CBPpredsA[i]))
    print(CBPpreds)
    results = pd.DataFrame()
    results.insert(0, 'CBP', CBPpreds)
    results.insert(1, 'Y', y_test)
    results = results.sort_values(by=['CBP'], ascending=False)
    results.to_csv(pred_dir + 'preds' + id + '.csv')
    np.save(pred_dir + 'preds' + id, np.array(CBPpreds))
    print(results.info())
    print(results.head(20))
    
    spearmanCBP = spearmanr(y_test, CBPpreds)[0]
    pearsonCBP = pearsonr(y_test, CBPpreds)[0]

    with open(pred_dir + 'corr' + id + 'txt', 'w') as f:
        f.write(id)
        f.write('\n')
        f.write('spearman CBP: ')
        f.write(str(spearmanCBP))
        f.write('\n')
        f.write('pearson CBP: ')
        f.write(str(pearsonCBP))
        f.write('\n')
        f.write('\n')
    # print("spearman CBP:", spearmanCBP)
    # print("pearson CBP:", pearsonCBP)