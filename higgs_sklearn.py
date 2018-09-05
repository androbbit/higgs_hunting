
#This script aims to study performance of [Decision Tree Classification with AdaBoost or GradientBoost] varing:
#-a choice of features (primary quantities (PRI) or derived quantities (DER))
#-the classifier parameter tuning (one at once)
#For cross validation, the provided training sampe is randomly split into two: one for train, one for test

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

#sklearn boosted tree classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#sklearn train/test split function
from sklearn.model_selection import train_test_split

# File system manangement
import os
import math

# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

#choose a type of feature to use and classifier    
feature_type = 'PRI'
classifier = 'ada'


def set_classifier(n_estimators=50, learning_rate = 1.0):
    if classifier = 'ada':
        clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    elif classifier = 'gb':
        clf =  GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    return clf

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

#Read the training file and discarding NaN entries
df = pd.read_csv('data/training.csv')
df.replace({-999.: np.nan}, inplace = True)
df = df.dropna()

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
        

train, test = train_test_split(df,test_size = 0.33)

if feature_type == 'PRI':
    features_selected = features_PRI
elif feature_type == 'DER':
    features_selected = features_DER
else:
    features_selected = features_ALL

#Split the features and label
X_train = train[features_selected]
y_train = train['Label']

X_val = test[features_selected]
y_val = test['Label']

#Define lists to save results for later-plotting
score_ne_cv = np.zeros(20)
score_ne_self = np.zeros(20)

ams_ne_cv = np.zeros(20)
ams_ne_self = np.zeros(20)

n_estimators = range(10,210,10)

#Vary number of estimators from 10 to 200
for i in range(20):
    clf = set_classifier(n_estimators=(i+1)*10)
    clf.fit(X_train, y_train)
    result_self = clf.predict(X_train)
    s_self = (result_self == 's')
    b_self = (result_self == 'b')

    result_cv = clf.predict(X_val)
    s_cv = (result_cv == 's')
    b_cv = (result_cv == 'b')

    score_ne_self[i] = clf.score(X_train,y_train)
    score_ne_cv[i] = clf.score(X_val,y_val)

    ams_ne_self[i] = AMS(s_self.sum(), b_self.sum())
    ams_ne_cv[i] = AMS(s_cv.sum(), b_cv.sum())

#Plot socres as a function of the number of estimators
plt.plot(n_estimators, score_ne_cv, label='Test Data')
plt.plot(n_estimators, score_ne_self,  label='Training Data')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.legend(loc='best')
plt.savefig('plots/results/score_vs_nestimators'+feature_type+'_'+classifier+'.png')
plt.close()

#Plot AMS as a function of the number of estimators
plt.plot(n_estimators, ams_ne_cv, label='Test Data')
plt.plot(n_estimators, ams_ne_self,  label='Training Data')
plt.xlabel('Number of Estimators')
plt.ylabel('AMS')
plt.legend(loc='best')
plt.savefig('plots/results/ams_vs_nestimators'+feature_type+'_'+classifier+'.png')
plt.close()

score_lr_cv = np.zeros(10)
score_lr_self = np.zeros(10)

ams_lr_cv = np.zeros(10)
ams_lr_self = np.zeros(10)

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#Vary learning rate from 0.1 to 1.0
for i in range(10):
    clf = set_classifier(learning_rate=(i+1)*0.1)
    clf.fit(X_train, y_train)

    result_self = clf.predict(X_train)
    s_self = (result_self == 's')
    b_self = (result_self == 'b')

    result_cv = clf.predict(X_val)
    s_cv = (result_cv == 's')
    b_cv = (result_cv == 'b')

    score_lr_self[i] = clf.score(X_train,y_train)
    score_lr_cv[i] = clf.score(X_val,y_val)

    ams_lr_self[i] = AMS(s_self.sum(), b_self.sum())
    ams_lr_cv[i] = AMS(s_cv.sum(), b_cv.sum())


#Plot socres as a function of the learning rate
plt.plot(learning_rates, score_lr_self, label='Training Data')
plt.plot(learning_rates, score_lr_cv, label='Test Data')
plt.xlabel('Learning Rate')
plt.ylabel('Score')
plt.legend(loc='best')
plt.savefig('plots/results/score_vs_learningrate'+feature_type+'_'+classifier+'.png')
plt.close()

#Plot AMS as a function of the learning rate
plt.plot(learning_rates, ams_lr_self, label='Training Data')
plt.plot(learning_rates, ams_lr_cv, label='Test Data')
plt.xlabel('Learning Rate')
plt.ylabel('AMS')
plt.legend(loc='best')
plt.savefig('plots/results/ams_vs_learningrate'+feature_type+'_'+classifier+'.png')
plt.close()
