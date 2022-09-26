import numpy as np
import pandas as pd
import os
import sys

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
from load_data import load_data, load_data_kampman
from sklearn.model_selection import train_test_split


def partition_data_abell(id, loaded_dir, part_dir):
    XR, XA, seqR, seqA, y = load_data(loaded_dir + id + '.csv')
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    np.save(part_dir + 'XR' + id, XR)
    np.save(part_dir + 'XA' + id, XA)
    np.save(part_dir + 'delta' + id, y)

    # XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, train_size=tts)
    # np.save('data/eval/XR_train' + id, XR_train)
    # np.save('data/eval/XA_train' + id, XA_train)
    # np.save('data/eval/XR_test' + id, XR_test)
    # np.save('data/eval/XA_test' + id, XA_test)
    # np.save('data/eval/y_train' + id, y_train)
    # np.save('data/eval/y_test' + id, y_test)

def partition_data_kampman(id, loaded_dir, part_dir):
    XR, XA, seqR, seqA, y = load_data_kampman(loaded_dir + id + '.csv')
    
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    np.save(part_dir + 'XR' + id, XR)
    np.save(part_dir + 'XA' + id, XA)
    np.save(part_dir + 'delta' + id, y)

    # XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, train_size=1)
    # np.save('data/MPRA_partitioned/KampmanV1/XR_train.peaks', XR_train)
    # np.save('data/MPRA_partitioned/KampmanV1/XA_train.peaks', XA_train)
    # np.save('data/MPRA_partitioned/KampmanV1/XR_test.peaks', XR_test)
    # np.save('data/MPRA_partitioned/KampmanV1/XA_test.peaks', XA_test)
    # np.save('data/MPRA_partitioned/KampmanV1/y_train.peaks', y_train)
    # np.save('data/MPRA_partitioned/KampmanV1/y_test.peaks', y_test)


