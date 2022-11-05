import numpy as np
import pandas as pd
import os
import sys
import seaborn
import matplotlib.pyplot as plt

os.chdir('/wynton/home/corces/allan/MPRAModel')
sys.path.append('/wynton/home/corces/allan/MPRAModel/utils')
sys.path.append('/wynton/home/corces/allan/MPRAModel/MPRA_data_analysis')

M1 = pd.read_csv('data/GWAS/Bellenguez/PlotSeabornGWAS.csv')

index = list(range(0, 10242))
M1 = M1.sort_values(by = ['M1jsd'], ascending=True)
M1['indexjsd'] = index
M1 = M1.sort_values(by = ['M1lfc'], ascending=True)
M1['indexlfc'] = index

dims = (10, 5)
fig, ax = plt.subplots(figsize=dims)
# seaborn.scatterplot(x="index", y="M1jsd", data=M1, size="M1jsd")
seaborn.scatterplot(x="indexjsd", y="M1jsd", data=M1, hue="SignifID", linewidth = 0, palette = "dark:#5A9_r", alpha=0.5, ax=ax)
# seaborn.scatterplot(x="indexlfc", y="M1lfc", data=M1, hue="SignifID", linewidth = 0, palette = "dark:#5A9_r", alpha=0.5, ax=ax, s = 50)
plt.savefig('figures/GWASM1jsd.svg')