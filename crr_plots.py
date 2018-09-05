#This script produces plots of correlations of the features to explore the data

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# File system manangement
import os

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/training.csv')

features_PRI = []
features_DER = []
features_ALL = []

features = df.keys()
features = features.delete(-1)
features = features.delete(-1)
features = features.delete(0)

df.replace({-999.: np.nan}, inplace = True)


for feature in features:
    if 'PRI' in feature:
        features_PRI.append(feature)
    elif 'DER' in feature:
        features_DER.append(feature)


df_PRI = df[features_PRI]
df_DER = df[features_DER]

df_PRI_corrs = df_PRI.corr()
df_DER_corrs = df_DER.corr()

zmin = df_PRI_corrs.min().min()
zmax = df_PRI_corrs.max().max()

plt.figure(figsize=(20,15))
sns.heatmap(df_PRI_corrs, cmap = plt.cm.RdYlBu_r, vmin = -1., annot = False, vmax = 1.)
plt.title('Correlation Heatmap');

figTitle = 'plots/Corr/correlation_PRI.png'
plt.savefig(figTitle)
plt.close()

plt.figure(figsize=(20,15))
sns.heatmap(df_DER_corrs, cmap = plt.cm.RdYlBu_r, vmin = -1., annot = False, vmax = 1.)
plt.title('Correlation Heatmap');

figTitle = 'plots/Corr/correlation_DER.png'
plt.savefig(figTitle)
plt.close()
