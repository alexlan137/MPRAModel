import numpy as np
import pandas as pd
import os
import sys
import pysam

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
from load_data import load_data, load_data_kampman, load_data_QTL, load_data_kampman_relu
from sklearn.model_selection import train_test_split
from mseqgen.sequtils import one_hot_encode



def partition_data_abell(id, loaded_dir, part_dir, tts):
    XR, XA, seqR, seqA, y = load_data(loaded_dir + id + '.csv')
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    np.save(part_dir + 'XR' + id, XR)
    np.save(part_dir + 'XA' + id, XA)
    np.save(part_dir + 'delta' + id, y)

    XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, test_size=tts)
    np.save(part_dir + 'XR_train' + id, XR_train)
    np.save(part_dir + 'XA_train' + id, XA_train)
    np.save(part_dir + 'XR_test' + id, XR_test)
    np.save(part_dir + 'XA_test' + id, XA_test)
    np.save(part_dir + 'delta_train' + id, y_train)
    np.save(part_dir + 'delta_test' + id, y_test)

def partition_data_kampman(id, loaded_dir, part_dir, tts):
    XR, XA, seqR, seqA, y = load_data_kampman(loaded_dir + id + '.csv')
    
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    np.save(part_dir + 'XR' + id, XR)
    np.save(part_dir + 'XA' + id, XA)
    np.save(part_dir + 'delta' + id, y)

    XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, test_size=tts)
    np.save(part_dir + 'XR_train' + id, XR_train)
    np.save(part_dir + 'XA_train' + id, XA_train)
    np.save(part_dir + 'XR_test' + id, XR_test)
    np.save(part_dir + 'XA_test' + id, XA_test)
    np.save(part_dir + 'delta_train' + id, y_train)
    np.save(part_dir + 'delta_test' + id, y_test)

def partition_data_kampman_relu(id, loaded_dir, part_dir, tts):
    XR, XA, seqR, seqA, y, flip = load_data_kampman_relu(loaded_dir + id + '.csv')
    
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    np.save(part_dir + 'XR' + id, XR)
    np.save(part_dir + 'XA' + id, XA)
    np.save(part_dir + 'delta' + id, y)
    np.save(part_dir + 'flip' + id, flip)

    XR_train, XR_test, XA_train, XA_test, y_train, y_test, flip_train, flip_test= train_test_split(XR, XA, y, flip, test_size=tts)
    np.save(part_dir + 'XR_train' + id, XR_train)
    np.save(part_dir + 'XA_train' + id, XA_train)
    np.save(part_dir + 'XR_test' + id, XR_test)
    np.save(part_dir + 'XA_test' + id, XA_test)
    np.save(part_dir + 'delta_train' + id, y_train)
    np.save(part_dir + 'delta_test' + id, y_test)
    np.save(part_dir + 'flip_train' + id, flip_train)
    np.save(part_dir + 'flip_test' + id, flip_test)

def insert_variant(seq, allele, position):
    """ Inserts the specified allele at the given position of the sequence
    """
    left, right = seq[:position-1], seq[position:]
    return left + allele + right

def partition_data_QTL(id, data_dir, part_dir, tts):
    XR, XA, seqR, seqA, y = load_data_QTL(data_dir + id + '.csv')
    print("xrshape:", XR.shape)
    print("xashape:", XA.shape)
    print("yshape:", y.shape)
    np.save(part_dir + 'XR' + id, XR)
    np.save(part_dir + 'XA' + id, XA)
    np.save(part_dir + 'delta' + id, y)

    XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR, XA, y, test_size=tts)
    np.save(part_dir + 'XR_train' + id, XR_train)
    np.save(part_dir + 'XA_train' + id, XA_train)
    np.save(part_dir + 'XR_test' + id, XR_test)
    np.save(part_dir + 'XA_test' + id, XA_test)
    np.save(part_dir + 'delta_train' + id, y_train)
    np.save(part_dir + 'delta_test' + id, y_test)

    
    # print("xrshape:", XR.shape)
    # print("xashape:", XA.shape)
    # print("yshape:", y.shape)
    # np.save(data_dir + 'XR' + id, XR)
    # np.save(data_dir + 'XA' + id, XA)
    # np.save(data_dir + 'delta' + id, y)

    # XR_subset, XR_trash, XA_subset, XR_trash, y_subset, y_trash = train_test_split(XR, XA, y, test_size=0.9)
    # XR_train, XR_test, XA_train, XA_test, y_train, y_test = train_test_split(XR_subset, XA_subset, y_subset, test_size=tts)
    # print("xrtr:", XR_train.shape)
    # print("xatr:", XA_train.shape)
    # print("ytrshape:", y_train.shape)
    # np.save(data_dir + 'XR' + id, XR_subset)
    # np.save(data_dir + 'XA' + id, XA_subset)
    # np.save(data_dir + 'delta' + id, y_subset)
    # np.save(data_dir + 'XR_train' + id, XR_train)
    # np.save(data_dir + 'XA_train' + id, XA_train)
    # np.save(data_dir + 'XR_test' + id, XR_test)
    # np.save(data_dir + 'XA_test' + id, XA_test)
    # np.save(data_dir + 'delta_train' + id, y_train)
    # np.save(data_dir + 'delta_test' + id, y_test)


