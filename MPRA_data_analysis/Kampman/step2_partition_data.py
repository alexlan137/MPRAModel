import numpy as np
import pandas as pd
import os
import sys

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
from load_data import load_data_kampman
from sklearn.model_selection import train_test_split


def partition_data():
    XR, XA, seqR, seqA, y = load_data_kampman('data/MPRA_loaded/Kampman.csv')
    
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    np.save('data/MPRA_partitioned/KampmanV1/XR', XR)
    np.save('data/MPRA_partitioned/KampmanV1/XA', XA)
    np.save('data/MPRA_partitioned/KampmanV1/delta', y)

    # XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, train_size=1)
    # np.save('data/eval/XR_train', XR_train)
    # np.save('data/eval/XA_train', XA_train)
    # np.save('data/eval/XR_test', XR_test)
    # np.save('data/eval/XA_test', XA_test)
    # np.save('data/eval/y_train', y_train)
    # np.save('data/eval/y_test', y_test)

if ('__main__'):
    partition_data()

