from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def graph_corr(id, pred_dir):
    CBP = pd.read_csv(pred_dir + 'preds' + id + '.csv')
    print(CBP.info())
    print(CBP.head())

    fig, ax = plt.subplots(figsize = (10, 7))
    CBP.plot.scatter(x='CBP', y='Y', c='DarkBlue', ax=ax)
    plt.savefig(pred_dir + 'corr' + id + '.png')
