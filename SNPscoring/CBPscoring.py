import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import pysam
from mseqgen.sequtils import one_hot_encode
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax


os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development')


SNPs = pd.read_csv('data/GWAS/Alzheimers_Bellenguez_2022.all_ld.bed', sep='\t')
print(SNPs.info())
print(SNPs.head())

data_dir = 'data/GWAS/Bellenguez/'
XA = np.load(data_dir + 'XA.npy')
XB = np.load(data_dir + 'XB.npy')

from load_model import load_CBPL

avail = ['1', '2', '5', '8', '13', '19', '24']
for modelid in avail:
    model = load_CBPL(modelid)
    predA_logits, predA_logcts = model.predict([XA,
                                              np.zeros((len(XA), model.output_shape[0][1])),
                                              np.zeros((len(XA), ))],
                                              batch_size=256, verbose=True)
    predB_logits, predB_logcts = model.predict([XB,
                                              np.zeros((len(XB), model.output_shape[0][1])),
                                              np.zeros((len(XB), ))],
                                              batch_size=256, verbose=True)
    print(len(predA_logits), len(predA_logits[0]))
    print(len(predB_logits), len(predA_logits[0]))
    print(len(predA_logcts))
    print(len(predB_logcts))
    SNPs['M' + modelid + 'lfc'] = np.log2(np.divide(np.exp(predA_logcts), np.exp(predB_logcts)))
    jsd = []
    for i in range(len(predA_logcts)):
        jsd.append(jensenshannon(softmax(predA_logits[i]), softmax(predB_logits[i])))
    SNPs['M' + modelid + 'jsd'] = jsd
    print(SNPs.head())
    print(SNPs.info())

SNPs.to_csv('data/GWAS/Bellenguez/scored_all.csv', sep='\t')