import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Lambda, Conv1D, Dense, Flatten, Cropping1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.callbacks as tfcallbacks 
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt


os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_model_development')

def flatten(l):
    return [item for sublist in l for item in sublist]

def train(cbpdir, datadir, dataid, mpramodelid, versionid, arch, lr):
    cbpdir = 'Soumya_K562'
    datadir = 'data/MPRA_partitioned/Kampman'
    mpramodelid = 'MOD2.Kampman.t1000.p0.5.c300'
    dataid = 'Kampman.mSK_K562.t100.p0.5.c300'
    version = '1'
    lr = 0.0001
    print("here")
    setupTNN(1, 'models/' + cbpdir + '/chrombpnet_wo_bias.h5', 0, 0, 0, 0, 0, 0)
    setupTNN(2, 'models/' + cbpdir + '/chrombpnet_wo_bias.h5', 0, 0, 0, 0, 0, 0)
    setupTNN(3, 'models/' + cbpdir + '/chrombpnet_wo_bias.h5', 0, 0, 0, 0, 0, 0)
    # TNN = setupTNN('models/' + cbpdir + '/chrombpnet_wo_bias.h5', lr) #MODIFY
    # print(TNN.summary())
    # XR_train = np.load(datadir + '/XR_train' + dataid + '.npy')
    # XA_train = np.load(datadir + '/XA_train' + dataid + '.npy')
    # y_train = np.load(datadir + '/delta_train' + dataid + '.npy')
    
    # print(XR_train.shape, XA_train.shape, y_train.shape)

    # checkpointer = tfcallbacks.ModelCheckpoint(filepath='MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + versionid, monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    # earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=5, verbose=1, min_delta=0.0005, restore_best_weights=True)
    # cur_callbacks=[checkpointer, earlystopper]
    # print("LR:", lr)
    # TNN.fit([XR_train, XA_train], y_train, batch_size=16, epochs=40, validation_split=0.2, callbacks=cur_callbacks)
    # TNN.save('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.v' + versionid + '.h5')

from load_model import load
from merge import merge

# type 1 = ChromBPNet; type 2 = ChromBPNet freeze; type 3 = ChromBPNet trimmed
def setupTNN(type, cbpfile, numconv, kernelconv, filterconv, numdense, filterdense, crop, lr):
    AlleleR = Input(shape=(2114, 4))
    AlleleA = Input(shape=(2114, 4))

    ChromBPNet = load(cbpfile)
    
    if (type == 0):
        EncodedR = ChromBPNet(inputs=[AlleleR])
        EncodedA = ChromBPNet(inputs=[AlleleA])

        EncodedR = Lambda(merge)(EncodedR)
        EncodedA = Lambda(merge)(EncodedA)
        Diff = Lambda(lambda tensors:keras.backend.log(tf.math.divide(tensors[0], tensors[1])))
        LFC = Diff([EncodedR, EncodedA])
        LFC = tf.expand_dims(LFC, axis=-1)
        X = Cropping1D(cropping=crop)(LFC)

        for i in range(numconv):
            X = Conv1D(filters=filterconv, kernel_size=kernelconv, padding='valid', activation='relu')(X)
    
        X = Flatten()(X)

        for i in range(numdense):
            X = Dense(filterdense, activation='relu')(X)
    
        prediction = Dense(1, activation='relu')(X)
        TNN = keras.Model(inputs=[AlleleR, AlleleA], outputs=[prediction])
        TNN.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
        return TNN

    if (type == 1):
        print("——TYPE 1——")
        print(ChromBPNet.summary())
        EncodedR = ChromBPNet(inputs=[AlleleR])
        EncodedA = ChromBPNet(inputs=[AlleleA])

        EncodedR = Lambda(merge)(EncodedR)
        EncodedA = Lambda(merge)(EncodedA)

    if (type == 2):
        ChromBPNet.trainable = False
        print("——TYPE 2——")
        print(ChromBPNet.summary())
        EncodedR = ChromBPNet(inputs=[AlleleR])
        EncodedA = ChromBPNet(inputs=[AlleleA])

        EncodedR = Lambda(merge)(EncodedR)
        EncodedA = Lambda(merge)(EncodedA)
    
    if (type == 3):
        ChromBPNet.layers.pop()
        ChromBPNet.layers.pop()
        ChromBPNet = Model(ChromBPNet.input, ChromBPNet.get_layer('wo_bias_bpnet_prof_out_precrop').output)
        print("——TYPE 3——")
        print(ChromBPNet.summary())
        EncodedR = ChromBPNet(inputs=[AlleleR])
        EncodedA = ChromBPNet(inputs=[AlleleA])
    
    

    Diff = Lambda(lambda tensors:keras.backend.log(tf.math.divide(tensors[0], tensors[1])))
    LFC = Diff([EncodedR, EncodedA])

    if (type == 1 or type == 2):
        LFC = tf.expand_dims(LFC, axis=-1)
    
    X = Cropping1D(cropping=crop)(LFC)

    for i in range(numconv):
        X = Conv1D(filters=filterconv, kernel_size=kernelconv, padding='valid', activation='relu')(X)
    
    X = Flatten()(X)

    for i in range(numdense):
        X = Dense(filterdense, activation='relu')(X)
    
    prediction = Dense(1, activation='tanh')(X)
    TNN = keras.Model(inputs=[AlleleR, AlleleA], outputs=[prediction])
    TNN.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    print(TNN.summary())
    return TNN

def trainTNN(mpramodelid, datadir, dataid, type, cbpfile, numconv, kernelconv, filterconv, numdense, filterdense, crop, lr):
    
    TNN = setupTNN(type, cbpfile, numconv, kernelconv, filterconv, numdense, filterdense, crop, lr)
    
    XR_train = np.load(datadir + '/XR_train' + dataid + '.npy')
    XA_train = np.load(datadir + '/XA_train' + dataid + '.npy')
    y_train = np.load(datadir + '/delta_train' + dataid + '.npy')

    checkpointer = tfcallbacks.ModelCheckpoint(filepath='MPRA_model_development/models/' + mpramodelid, monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=5, verbose=1, min_delta=0.0001, restore_best_weights=True)
    cur_callbacks=[checkpointer, earlystopper]
    TNN.fit([XR_train, XA_train], y_train, batch_size=16, epochs=40, validation_split=0.2, callbacks=cur_callbacks)
    TNN.save('MPRA_model_development/models/MPRAModel.' + mpramodelid + '.h5')

def evalTNN(mpramodelid, datadir, dataid):
    TNN = tf.keras.models.load_model('MPRA_model_development/models/' + mpramodelid)
    print(TNN.summary())

    XR = np.load(datadir + '/XR' + dataid + '.npy')
    XA = np.load(datadir + '/XA' + dataid + '.npy')
    y = np.load(datadir + '/delta' + dataid + '.npy')
    flip = np.load(datadir + '/flip' + dataid + '.npy')

    XR_train = np.load(datadir + '/XR_train' + dataid + '.npy')
    XA_train = np.load(datadir + '/XA_train' + dataid + '.npy')
    y_train = np.load(datadir + '/delta_train' + dataid + '.npy')
    flip_train = np.load(datadir + '/flip_train' + dataid + '.npy')

    XR_test = np.load(datadir + '/XR_test' + dataid + '.npy')
    XA_test = np.load(datadir + '/XA_test' + dataid + '.npy')
    y_test = np.load(datadir + '/delta_test' + dataid + '.npy')
    flip_test = np.load(datadir + '/flip_test' + dataid + '.npy')

    TNNpreds = TNN.predict([XR, XA], batch_size=16, verbose=True)
    np.save('MPRA_model_development/models/' + mpramodelid + '/preds', np.array(TNNpreds))
    TNNpreds = flatten(TNNpreds)
    print(TNNpreds)
    print(y)
    resultsv1 = pd.DataFrame()
    resultsv1['TNN'] = TNNpreds
    resultsv1['Y'] = y
    resultsv1['flip'] = flip
    results = pd.DataFrame()
    results['TNN'] = resultsv1['TNN'] * resultsv1['flip']
    results['Y'] = resultsv1['Y'] * resultsv1['flip']
    # results = pd.DataFrame()
    # results['TNN'] = TNNpreds
    # results['Y'] = y
    results.sort_values(by=['Y'])
    print(results.info())
    print(results.head(20))

    TNNpreds_train = TNN.predict([XR_train, XA_train], batch_size=16, verbose=True)
    np.save('MPRA_model_development/models/' + mpramodelid + '/preds.train', np.array(TNNpreds_train))
    TNNpreds_train = flatten(TNNpreds_train)
    print(TNNpreds_train)
    print(y_train)
    resultsv1_train = pd.DataFrame()
    resultsv1_train['TNN'] = TNNpreds_train
    resultsv1_train['Y'] = y_train
    resultsv1_train['flip'] = flip_train
    results_train = pd.DataFrame()
    results_train['TNN'] = resultsv1_train['TNN'] * resultsv1_train['flip']
    results_train['Y'] = resultsv1_train['Y'] * resultsv1_train['flip']
    # results_train = pd.DataFrame()
    # results_train['TNN'] = TNNpreds_train
    # results_train['Y'] = y_train
    results_train.sort_values(by=['Y'])
    print(results_train.info())
    print(results_train.head(20))

    TNNpreds_test = TNN.predict([XR_test, XA_test], batch_size=16, verbose=True)
    np.save('MPRA_model_development/models/' + mpramodelid + '/preds.test', np.array(TNNpreds_test))
    TNNpreds_test = flatten(TNNpreds_test)
    print(TNNpreds_test)
    print(y_test)
    resultsv1_test = pd.DataFrame()
    resultsv1_test['TNN'] = TNNpreds_test
    resultsv1_test['Y'] = y_test
    resultsv1_test['flip'] = flip_test
    results_test = pd.DataFrame()
    results_test['TNN'] = resultsv1_test['TNN'] * resultsv1_test['flip']
    results_test['Y'] = resultsv1_test['Y'] * resultsv1_test['flip']
    # results_test = pd.DataFrame()
    # results_test['TNN'] = TNNpreds_test
    # results_test['Y'] = y_test
    results_test.sort_values(by=['Y'])
    print(results_test.info())
    print(results_test.head(20))


    spearmanTNN = spearmanr(np.array(results['Y']), np.array(results['TNN']))[0]
    pearsonTNN = pearsonr(np.array(results['Y']), np.array(results['TNN']))[0]
    spearmanTNN_train = spearmanr(np.array(results_train['Y']), np.array(results_train['TNN']))[0]
    pearsonTNN_train = pearsonr(np.array(results_train['Y']), np.array(results_train['TNN']))[0]
    spearmanTNN_test = spearmanr(np.array(results_test['Y']), np.array(results_test['TNN']))[0]
    pearsonTNN_test = pearsonr(np.array(results_test['Y']), np.array(results_test['TNN']))[0]
    
    print("spearman TNN-all:", spearmanTNN)
    print("pearson TNN-all:", pearsonTNN)
    print("spearman TNN-train:", spearmanTNN_train)
    print("pearson TNN-train:", pearsonTNN_train)
    print("spearman TNN-test:", spearmanTNN_test)
    print("pearson TNN-test:", pearsonTNN_test)

    fig, ax = plt.subplots(figsize = (10, 10))
    _, bins, _ = ax.hist(np.array(results_train['Y']), bins=50, color='coral', alpha=0.5, label = 'mpratrain')
    ax.hist(np.array(results_test['Y']), bins=bins, color='coral', alpha=0.5, label = 'mpratest')
    ax.hist(np.array(results_train['TNN']), bins=bins, color='darkblue', alpha=0.5, label = 'preds_train')
    ax.hist(np.array(results_test['TNN']), bins=bins, color='black', alpha=0.5, label = 'preds_test')
    plt.legend()
    plt.savefig('MPRA_model_development/models/' + mpramodelid + '/distribution.all.png')

    fig, ax = plt.subplots(figsize = (10, 10))
    results_train.plot.scatter(x='TNN', y='Y', c='darkblue', alpha=0.8, ax=ax)
    results_test.plot.scatter(x='TNN', y='Y', c='red', alpha=0.8, ax=ax)
    plt.savefig('MPRA_model_development/models/' + mpramodelid + '/correlation.all.png')

    with open('MPRA_model_development/models/' + mpramodelid + '/metrics.txt', 'w') as f:
        f.write('spearman TNN-all: ')
        f.write(str(spearmanTNN))
        f.write('\n')
        f.write('pearson TNN-all: ')
        f.write(str(pearsonTNN))
        f.write('\n')
        f.write('spearman TNN-train: ')
        f.write(str(spearmanTNN_train))
        f.write('\n')
        f.write('pearson TNN-train: ')
        f.write(str(pearsonTNN_train))
        f.write('\n')
        f.write('spearman TNN-test: ')
        f.write(str(spearmanTNN_test))
        f.write('\n')
        f.write('pearson TNN-test: ')
        f.write(str(pearsonTNN_test))
        f.write('\n')

def mainTNN():
    index = 1
    type = 0
    cbpfile = 'models/Soumya_K562/chrombpnet_wo_bias.h5'
    datadir = 'data/MPRA_partitioned/KampmanRelu'
    dataid = 'KampmanRelu.mSK_K562.t100.p0.7.c300'
    
    crop = 0
    numconv = [0, 1, 2]
    kernelconv = [3, 5, 8]
    filterconv = [16, 32]
    numdense = [1]
    filterdense = [48]

    for convs in numconv:

        mpramodelid = 'STS' + str(index)
        numconv = convs
        kernelconv = 8
        filterconv = 16
        numdense = 1
        filterdense = 64
        lr = 0.000005
        
        with open('MPRA_model_development/models/' + mpramodelid + '/hyperparams.txt', 'w') as f:
            f.write('type: ')
            f.write(str(type))
            f.write('\n')
            f.write('cell type: K562')
            f.write('\n')
            f.write('numconv: ')
            f.write(str(numconv))
            f.write('\n')
            f.write('kernelconv: ')
            f.write(str(kernelconv))
            f.write('\n')
            f.write('filterconv: ')
            f.write(str(filterconv))
            f.write('\n')
            f.write('numdense: ')
            f.write(str(numdense))
            f.write('\n')
            f.write('filterdense: ')
            f.write(str(filterdense))
            f.write('\n')
            f.write('crop: ')
            f.write(str(crop))
            f.write('\n')
            f.write('lr: ')
            f.write(str(lr))
            f.write('\n')
            f.write('dataid: ')
            f.write(dataid)
        
        trainTNN(mpramodelid, datadir, dataid, type, cbpfile, numconv, kernelconv, filterconv, numdense, filterdense, crop, lr)
        evalTNN(mpramodelid, datadir, dataid)
        index = index + 1

if('__main__'):
    mainTNN()