import numpy as np
import pandas as pd
import os
import sys

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_data_analysis')

MODEL_AL = '/wynton/home/corces/allan/MPRAModel/models/GM12878/chrombpnet_wo_bias.h5'
MODEL_SK = '/wynton/home/corces/allan/MPRAModel/models/Soumya_GM12878/chrombpnet_wo_bias.h5'
MODEL_K562 = '/wynton/home/corces/allan/MPRAModel/models/Soumya_K562/chrombpnet_wo_bias.h5'

# PARAMETERS
DATASET = 'KampmanRelu'
PLASMID_THRESHOLD = 100
CDNA_THRESHOLD = 100
PVAL = 0.7
LFC_CROP = 300
CUR_MODEL = "SK_K562"
MODEL_LOC = MODEL_SK
TRAIN_TEST_SPLIT = 0.2 # test_size


MPRA_DIR = '/wynton/home/corces/allan/MPRAModel/data/MPRA/' + DATASET + '/'
MPRA_LOADED_DIR = '/wynton/home/corces/allan/MPRAModel/data/MPRA_loaded/' + DATASET + '/'
MPRA_PARTITION_DIR = '/wynton/home/corces/allan/MPRAModel/data/MPRA_partitioned/' + DATASET + '/'
PREDS_DIR = '/wynton/home/corces/allan/MPRAModel/predictions/' + DATASET + '/'
MPRA_CSV = MPRA_DIR + DATASET + '.csv'


ID = DATASET + '.m' + str(CUR_MODEL) + '.t' + str(PLASMID_THRESHOLD) + '.p' + str(PVAL) + '.c' + str(LFC_CROP)


from step1_loadAbell import loadAbell
from step1_loadKampman import loadKampman, loadKampmanRelu
from step1_loadAbellK import loadAbellK, loadAbellKScaled
from step2_partition_data import partition_data_abell, partition_data_kampman, partition_data_kampman_relu
from step3_eval_cbp import eval_cbp
from step4_distribution import graph_dist
from step5_correlation import graph_corr

if (DATASET == 'Kampman'):
    loadKampman(MPRA_CSV, PVAL, PLASMID_THRESHOLD, ID, MPRA_LOADED_DIR)
    partition_data_kampman(ID, MPRA_LOADED_DIR, MPRA_PARTITION_DIR, TRAIN_TEST_SPLIT)

if (DATASET == 'KampmanRelu'):
    loadKampmanRelu(MPRA_CSV, PVAL, PLASMID_THRESHOLD, ID, MPRA_LOADED_DIR)
    partition_data_kampman_relu(ID, MPRA_LOADED_DIR, MPRA_PARTITION_DIR, TRAIN_TEST_SPLIT)

if (DATASET == 'AbellK'):
    loadAbellK(MPRA_CSV, PVAL, PLASMID_THRESHOLD, ID, MPRA_LOADED_DIR)
    partition_data_kampman(ID, MPRA_LOADED_DIR, MPRA_PARTITION_DIR, TRAIN_TEST_SPLIT)

if (DATASET == 'AbellKScaled'):
    loadAbellKScaled(MPRA_CSV, PVAL, PLASMID_THRESHOLD, ID, MPRA_LOADED_DIR)
    partition_data_kampman(ID, MPRA_LOADED_DIR, MPRA_PARTITION_DIR, TRAIN_TEST_SPLIT)

if (DATASET == 'Abell'):
    loadAbell(MPRA_CSV, PVAL, PLASMID_THRESHOLD, CDNA_THRESHOLD, ID, MPRA_LOADED_DIR)
    partition_data_abell(ID, MPRA_LOADED_DIR, MPRA_PARTITION_DIR, TRAIN_TEST_SPLIT)

eval_cbp(MODEL_LOC, LFC_CROP, ID, MPRA_PARTITION_DIR, PREDS_DIR)
graph_dist(ID, MPRA_PARTITION_DIR, PREDS_DIR)
graph_corr(ID, PREDS_DIR)
