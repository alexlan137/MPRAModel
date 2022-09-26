from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def graph_dist(id, part_dir, pred_dir):
    delta = np.load(part_dir + 'delta' + id + '.npy')
    CBPpreds = np.load(pred_dir + 'preds' + id + '.npy')
    CBPpreds = np.reshape(CBPpreds, len(delta))
    CBPpreds = np.sort(CBPpreds)

    fig, ax = plt.subplots(figsize = (10, 10))
    _, bins, _ = ax.hist(delta, bins=50, color='coral', alpha=0.3, label = 'mpra')
    ax.hist(CBPpreds, bins=bins, color='darkblue', alpha=0.3, label = 'preds')
    plt.legend()
    plt.savefig(pred_dir + 'dist' + id + '.png')
