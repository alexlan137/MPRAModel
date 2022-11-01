import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development')

from step1_train_mpra import train_mpra
from step2_eval_mpra import eval_mpra
from step3_shap_TNN import shap_TNN

def run():
    # Three model directories: 'GM12878', 'Soumya_GM12878', 'Soumya_K562'
    
    cbpdir = 'GM12878'
    datadir = 'data/QTL'
    mpramodelid = 'QTL.mAL_GM12878'
    dataid = '1000GV2tenth'
    version = '2'
    lr = 0.00005

    # cbpdir = 'Soumya_K562'
    # datadir = 'data/MPRA_partitioned/Kampman'
    # mpramodelid = 'MOD2.Kampman.t1000.p0.5.c300'
    # dataid = 'Kampman.mSK_K562.t100.p0.5.c300'
    # version = '1'
    # lr = 0.0001

    train_mpra(cbpdir, datadir, dataid, mpramodelid, version, lr)
    
    with open('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + version + '/params.txt', 'w') as f:
        f.write('model: ')
        f.write(cbpdir)
        f.write('\n')
        f.write('dataset: ')
        f.write(datadir)
        f.write('\n')
        f.write('modelID: ')
        f.write(mpramodelid)
        f.write('\n')
        f.write('version: ')
        f.write(version)
        f.write('\n')
        f.write('learning rate: ')
        f.write(str(lr))
        f.write('\n')
        f.write('\n')
    eval_mpra(mpramodelid, datadir, dataid, version)
    # shap_TNN(mpramodelid, version, datadir, dataid)

if('__main__'):
    run()