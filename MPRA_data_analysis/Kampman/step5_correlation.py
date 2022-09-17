from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


CBP = pd.read_csv('MPRA_data_analysis/Kampman/KampmanV1CBPmetrics/preds_GM12878.csv')
print(CBP.info())
print(CBP.head())

fig, ax = plt.subplots(figsize = (10, 7))
CBP.plot.scatter(x='CBP', y='Y', c='DarkBlue', ax=ax)
plt.savefig('MPRA_data_analysis/Kampman/KampmanV1CBPmetrics/preds_GM12878.corr.png')
