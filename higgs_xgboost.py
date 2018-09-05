
#This script aims to study performance of [Decision Tree Classification with XGBoost] varing:
#-a choice of features (primary quantities (PRI) or derived quantities (DER))
#-the classifier parameter tuning (one at once)
#For cross validation, the provided training sampe is randomly split into two: one for train, one for test


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn train/test split function
from sklearn.model_selection import train_test_split

#XGBoost Decision Tree
import xgboost as xgb

# File system manangement
import os
import math

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

feature_type = 'PRI'

#Evaluation metric to be maximized 
def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    ams = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if ams < 0:
        print('ams is negative. Exiting')
        exit()
    else:
        return math.sqrt(ams)


#Read the data file
df = pd.read_csv('data/training.csv')
#df.replace({-999.: np.nan}, inplace = True)


#Create feature lists separated into its type: primary quantities (PRI) and derived quantities (DER)
features_PRI = []
features_DER = []
features_ALL = []

features = df.keys()
features = features.delete(0) #delete EventId
features = features.delete(-1) #delete Label
features = features.delete(1) #delete Weight
features_ALL = features

for feature in features:
    if 'PRI' in feature:
        features_PRI.append(feature)
    elif 'DER' in feature:
        features_DER.append(feature)
        
#Split the data: one for train, one for test
train, test = train_test_split(df,test_size = 0.33)

#Select the feature type to use for the training
if feature_type == 'PRI':
    features_selected = features_PRI
elif feature_type == 'DER':
    features_selected = features_DER
else:
    features_selected = features_ALL

X_train = train[features_selected]
y_train, uniques_train = pd.factorize(train['Label'])
w = train['Weight']

X_val = test[features_selected]
y_val, uniques_val = pd.factorize(test['Label'])

#Construct DMatrix from pandas dataframes to plug-in to XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_val, label=y_val)

#Set the classifier parameters
#max_depth: maximum depth of a tree. defualt = 6
#eta: learning range. default = 1. range = [0,1]
#silent: printing running messages (0) or not (1)
#objective: objective function to minimize 
#num_round: number of rounds for boosting
param = {'max_depth':1, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2

#Train the model
bst = xgb.train(param, dtrain, num_round)

# make prediction
#y_pred = bst.predict(dtest)

#Plotting and save to files
#xgb.plot_importance(bst)
#plt.show()
#plt.savefig('test1.png')
#plt.close()
xgb.plot_tree(bst)
plt.show()
#plt.savefig('test2.png')
#plt.close()
