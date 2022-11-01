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
DATASET = '1000GV2tenth'
LFC_CROP = 100
CUR_MODEL = "SK_GM12878"
MODEL_LOC = MODEL_AL

DATA_DIR = '/wynton/home/corces/allan/MPRAModel/data/QTL/'
PREDS_DIR = '/wynton/home/corces/allan/MPRAModel/predictions/' + DATASET + '/'

from step2_partition_data import partition_data_QTL
from step3_eval_cbp import eval_cbp
from step4_distribution import graph_dist
from step5_correlation import graph_corr


partition_data_QTL(DATASET, DATA_DIR, DATA_DIR, 0.2)
eval_cbp(MODEL_LOC, LFC_CROP, DATASET, DATA_DIR, PREDS_DIR)
graph_dist(DATASET, DATA_DIR, PREDS_DIR)
graph_corr(DATASET, PREDS_DIR)