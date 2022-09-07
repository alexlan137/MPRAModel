from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data_df = pd.read_csv('data/MPRA/train-abell-filtered-lfc.csv')
scores = np.array(data_df['delta'])
y_train = np.load('data/training/y_train.npy')
y_test = np.load('data/training/y_test.npy')
CBPpreds = np.load('metrics/cbppredsjsd.npy')
CBPpreds = np.reshape(CBPpreds, 4204)
CBPpreds = np.sort(CBPpreds)
TNNpreds = np.load('metrics/TNNpreds.npy')
sortedTNN = np.reshape(TNNpreds, 4204)
sortedTNN = np.sort(sortedTNN)
# print(sortedTNN)

print(len(TNNpreds), len(CBPpreds))
fig, ax = plt.subplots(figsize = (10, 7))
ax.hist(scores, bins=500, color='black')
ax.hist(y_train, bins=500, color='steelblue')
ax.hist(y_test, bins=500, color='steelblue') #coral
# ax.hist(CBPpreds, bins=500, color='red')
ax.hist(TNNpreds, bins=500, color='red')

plt.savefig('data/TNNpred-distribution.png')
