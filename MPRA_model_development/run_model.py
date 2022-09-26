import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development')

from train_mpra import train_mpra
from eval_mpra import eval_mpra

def run():
    # Three model directories: 'GM12878', 'Soumya_GM12878', 'Soumya_K562'
    cbpdir = 'Soumya_K562'
    datadir = 'data/MPRA_partitioned/Kampman'
    mpramodelid = 'Kampman.mSK_K562.t100.p0.5.c300'
    dataid = 'Kampman.mAL.t100.p0.5.c300'
    version = '1'
    lr = 0.0002
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

if('__main__'):
    run()