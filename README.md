# Higgs Hunting
## Identify events with Higgs involved from any other backgrounds (binary classification).

This branch contains preliminary plots and scripts to explore the data and study various classifier with parameter tuning. 

- plots/KDE: KDE plots for each feature
- plots/Corr: correlation matrices between the features

- kde_plots.py: script to produce plots in plots/KDE
- crr_plots.py: script to produce plots in plots/Corr
- higgs_sklearn.py: script to train the data with AdaBoostClassifier and GradientBoostingClassifier
- higss_xgboost.py: script to train the data with XGBoost
