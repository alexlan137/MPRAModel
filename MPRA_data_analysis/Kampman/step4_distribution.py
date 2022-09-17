from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data_df = pd.read_csv('data/MPRA_loaded/Kampman.csv')
scores = np.array(data_df['delta'])
# y_train = np.load('data/training/y_train.npy')
y_test = np.load('data/MPRA_partitioned/KampmanV1/delta.npy')
CBPpreds = np.load('MPRA_data_analysis/Kampman/KampmanV1CBPmetrics/preds_GM12878.npy')
CBPpreds = np.reshape(CBPpreds, 5255)
CBPpreds = np.sort(CBPpreds)
# TNNpreds = np.load('metrics/TNNpreds.npy')
# sortedTNN = np.reshape(TNNpreds, 4204)
# sortedTNN = np.sort(sortedTNN)
# print(sortedTNN)

# print(len(TNNpreds), len(CBPpreds))
fig, ax = plt.subplots(figsize = (10, 7))
# ax.hist(scores, bins=500, color='black')
# ax.hist(y_train, bins=500, color='steelblue')
ax.hist(y_test, bins=500, color='coral', alpha = 0.5) #coral
ax.hist(CBPpreds, bins=500, color='steelblue', alpha = 0.5)
# ax.hist(TNNpreds, bins=500, color='red')

plt.savefig('MPRA_data_analysis/Kampman/KampmanV1CBPmetrics/preds_GM12878.png')
