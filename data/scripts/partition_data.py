import numpy as np
import pandas as pd
import os
import sys

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
from load_data import load_data
from sklearn.model_selection import train_test_split


def partition_data():
    XR, XA, seqR, seqA, y = load_data('data/MPRA/train-abell-filtered.csv')
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, train_size=0.9)
    np.save('data/training/XR_train', XR_train)
    np.save('data/training/XA_train', XA_train)
    np.save('data/training/XR_test', XR_test)
    np.save('data/training/XA_test', XA_test)
    np.save('data/training/y_train', y_train)
    np.save('data/training/y_test', y_test)

if ('__main__'):
    partition_data()

