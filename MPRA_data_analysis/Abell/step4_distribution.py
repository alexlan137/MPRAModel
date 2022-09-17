from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data_df = pd.read_csv('data/MPRA_loaded/AbellV2.csv')
scores = np.array(data_df['delta'])
# y_train = np.load('data/training/y_train.npy')
y_test = np.load('MPRA_data_analysis/Abell/AbellV2_CBPmetrics_pval/delta.001.npy')
CBPpreds = np.load('MPRA_data_analysis/Abell/AbellV2_CBPmetrics_pval/pval0.001_Soumya_QNB.npy')
CBPpreds = np.reshape(CBPpreds, 455)
CBPpreds = np.sort(CBPpreds)
# TNNpreds = np.load('metrics/TNNpreds.npy')
# sortedTNN = np.reshape(TNNpreds, 4204)
# sortedTNN = np.sort(sortedTNN)
# print(sortedTNN)

# print(len(TNNpreds), len(CBPpreds))
fig, ax = plt.subplots(figsize = (10, 7))
# ax.hist(scores, bins=500, color='black')
# ax.hist(y_train, bins=500, color='steelblue')
ax.hist(y_test, bins=500, color='coral') #coral
ax.hist(CBPpreds, bins=500, color='black')
# ax.hist(TNNpreds, bins=500, color='red')

plt.savefig('MPRA_data_analysis/Abell/AbellV2_CBPmetrics_pval/pval.001_Soumya_QNB.png')
