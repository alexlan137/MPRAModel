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

    # XR = np.load(part_dir + 'XR' + id + '.npy')
    # XA = np.load(part_dir + 'XA' + id + '.npy')
    # y = np.load(part_dir + 'delta' + id + '.npy')

    # XR_train = np.load(part_dir + 'XR_train' + id + '.npy')
    # XA_train = np.load(part_dir + 'XA_train' + id + '.npy')
    # y_train = np.load(part_dir + 'delta_train' + id + '.npy')

    # CBPpredsR = CBP.predict([XR], verbose=True, batch_size=128)
    # CBPpredsR_logits = CBPpredsR[:][0]
    # CBPpredsR_logcts = CBPpredsR[:][1]
    # CBPpredsA = CBP.predict([XA], verbose=True, batch_size=128)
    # CBPpredsA_logits = CBPpredsA[:][0]
    # CBPpredsA_logcts = CBPpredsA[:][1]

    # CBPpredsR_train = CBP.predict([XR_train], verbose=True, batch_size=64)
    # CBPpredsR_train_logits = CBPpredsR_train[:][0]
    # CBPpredsR_train_logcts = CBPpredsR_train[:][1]
    # CBPpredsA_train = CBP.predict([XA_train], verbose=True, batch_size=64)
    # CBPpredsA_train_logits = CBPpredsA_train[:][0]
    # CBPpredsA_train_logcts = CBPpredsA_train[:][1]

    XR_test = np.load(part_dir + 'XR_test' + id + '.npy')
    XA_test = np.load(part_dir + 'XA_test' + id + '.npy')
    print(XR_test[0][1055:1059], XA_test[0][1055:1059])
    y_test = np.load(part_dir + 'delta_test' + id + '.npy')


    CBPpredsR_test = CBP.predict([XR_test], verbose=True, batch_size=64)
    CBPpredsR_test_logits = CBPpredsR_test[:][0]
    CBPpredsR_test_logcts = CBPpredsR_test[:][1]
    CBPpredsA_test = CBP.predict([XA_test], verbose=True, batch_size=64)
    CBPpredsA_test_logits = CBPpredsA_test[:][0]
    CBPpredsA_test_logcts = CBPpredsA_test[:][1]
    
    # CBPpreds = []
    # for i in range(len(CBPpredsR_logits)):
    #     CBPpredsR_profile = softmax(CBPpredsR_logits[i]) * (np.exp(CBPpredsR_logcts[i]) - 1)
    #     CBPpredsA_profile = softmax(CBPpredsA_logits[i]) * (np.exp(CBPpredsA_logcts[i]) - 1)
    #     track, lfc = log_fold_change(CBPpredsR_profile, CBPpredsA_profile, int(500 - (crop / 2)), int(500 + (crop / 2)))
    #     CBPpreds.append(lfc)
    # print(CBPpreds)

    # CBPpreds_train = []
    # for i in range(len(CBPpredsR_train_logits)):
    #     CBPpredsR_train_profile = softmax(CBPpredsR_train_logits[i]) * (np.exp(CBPpredsR_train_logcts[i]) - 1)
    #     CBPpredsA_train_profile = softmax(CBPpredsA_train_logits[i]) * (np.exp(CBPpredsA_train_logcts[i]) - 1)
    #     track_train, lfc_train = log_fold_change(CBPpredsR_train_profile, CBPpredsA_train_profile, int(500 - (crop / 2)), int(500 + (crop / 2)))
    #     CBPpreds_train.append(lfc_train)
    # print(CBPpreds_train)

    CBPpreds_test = []
    for i in range(len(CBPpredsR_test_logits)):
        CBPpredsR_test_profile = softmax(CBPpredsR_test_logits[i]) * (np.exp(CBPpredsR_test_logcts[i]) - 1)
        CBPpredsA_test_profile = softmax(CBPpredsA_test_logits[i]) * (np.exp(CBPpredsA_test_logcts[i]) - 1)
        track_test, lfc_test = log_fold_change(CBPpredsR_test_profile, CBPpredsA_test_profile, int(500 - (crop / 2)), int(500 + (crop / 2)))
        CBPpreds_test.append(lfc_test)
    print(CBPpreds_test)

    # results = pd.DataFrame()
    # results.insert(0, 'CBP', CBPpreds)
    # results.insert(1, 'Y', y)
    # results = results.sort_values(by=['CBP'], ascending=False)
    # results.to_csv(pred_dir + 'preds' + id + '.csv')
    # np.save(pred_dir + 'preds' + id, np.array(CBPpreds))
    # print(results.info())
    # print(results.head(20))

    # results_train = pd.DataFrame()
    # results_train.insert(0, 'CBP', CBPpreds_train)
    # results_train.insert(1, 'Y', y_train)
    # results_train = results_train.sort_values(by=['CBP'], ascending=False)
    # results_train.to_csv(pred_dir + 'preds_train' + id + '.csv')
    # np.save(pred_dir + 'preds_train' + id, np.array(CBPpreds_train))
    # print(results_train.info())
    # print(results_train.head(20))

    results_test = pd.DataFrame()
    results_test.insert(0, 'CBP', CBPpreds_test)
    results_test.insert(1, 'Y', y_test)
    results_test = results_test.sort_values(by=['CBP'], ascending=False)
    results_test.to_csv(pred_dir + 'preds_test' + id + '.csv')
    np.save(pred_dir + 'preds_test' + id, np.array(CBPpreds_test))
    print(results_test.info())
    print(results_test.head(20))
    
    # spearmanCBP = spearmanr(y, CBPpreds)[0]
    # pearsonCBP = pearsonr(y, CBPpreds)[0]
    
    # spearmanCBP_train = spearmanr(y_train, CBPpreds_train)[0]
    # pearsonCBP_train = pearsonr(y_train, CBPpreds_train)[0]

    spearmanCBP_test = spearmanr(y_test, CBPpreds_test)[0]
    pearsonCBP_test = pearsonr(y_test, CBPpreds_test)[0]

    with open(pred_dir + 'corr' + id + '.txt', 'w') as f:
        f.write(id)
        f.write('\n')
        # f.write('spearman CBP: ')
        # f.write(str(spearmanCBP))
        # f.write('\n')
        # f.write('pearson CBP: ')
        # f.write(str(pearsonCBP))
        # f.write('\n')
        # f.write('\n')
        # f.write('spearman CBP-train: ')
        # f.write(str(spearmanCBP_train))
        # f.write('\n')
        # f.write('pearson CBP-train: ')
        # f.write(str(pearsonCBP_train))
        # f.write('\n')
        # f.write('\n')
        f.write('spearman CBP-test: ')
        f.write(str(spearmanCBP_test))
        f.write('\n')
        f.write('pearson CBP-test: ')
        f.write(str(pearsonCBP_test))
        f.write('\n')
        f.write('\n')

# if('__main__'):
#     eval_cbp('models/Soumya_K562/chrombpnet_wo_bias.h5', 300, 'Kampman.mAL.t100.p0.5.c300', 'data/MPRA_partitioned/Kampman/', '/wynton/home/corces/allan/MPRAModel/predictions/Kampman/')