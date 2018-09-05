#This script produces plots of features separated in signal/background to explore the data


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# File system manangement
import os

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

#Read the data file
df = pd.read_csv('data/training.csv')

features_PRI = []
features_DER = []
features_ALL = []

features = df.keys()
features = features.delete(-1) #delete Label
features = features.delete(-1) #delelte Weight
features = features.delete(0) #delete Event Id

#Drop NaN value entries which might skew the distributions
df.replace({-999.: np.nan}, inplace = True)

for feature in features:
    if 'PRI' in feature:
        features_PRI.append(feature)
    elif 'DER' in feature:
        features_DER.append(feature)

    col_list = ['Label']
    col_list.append(feature)

    plot_data = df[col_list]
    plot_data = plot_data.dropna()

    sns.kdeplot(plot_data.loc[plot_data['Label'] == 's' , feature], label = 'Signal')
    sns.kdeplot(plot_data.loc[plot_data['Label'] == 'b' , feature], label = 'Background')
    plt.xlabel(feature)
    plt.ylabel('Density')
    figTitle = 'plots/KDE/'+feature+'.png'
    plt.savefig(figTitle)
    plt.close()

