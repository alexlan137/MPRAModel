from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def graph_corr(id, pred_dir):
    # CBP = pd.read_csv(pred_dir + 'preds' + id + '.csv')
    # CBP_train = pd.read_csv(pred_dir + 'preds_train' + id + '.csv')
    CBP_test = pd.read_csv(pred_dir + 'preds_test' + id + '.csv')
    # print(CBP.info())
    # print(CBP.head())
    # print(CBP_train.info())
    # print(CBP.head())
    print(CBP_test.info())
    # print(CBP.head())

    fig, ax = plt.subplots(figsize = (10, 10))
    # CBP_train.plot.scatter(x='CBP', y='Y', c='DarkBlue', ax=ax)
    CBP_test.plot.scatter(x='CBP', y='Y', c='Red', ax=ax)
    plt.savefig(pred_dir + 'corr' + id + '.png')

# if('__main__'):
#     graph_corr('Kampman.mAL.t100.p0.5.c300', '/wynton/home/corces/allan/MPRAModel/predictions/Kampman/')
