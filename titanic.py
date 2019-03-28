#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:18:28 2018

@author: crohr
"""

import pandas as pd
import numpy as np

import statsmodels.api as sm

from sklearn.metrics import classification_report

from data_preprocessing import get_features_labels
from data_preprocessing import get_training_testing
from data_preprocessing import scale_data
from data_preprocessing import correlation_heatmap
from analysis import perform_analysis


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
datafile = "/home/crohr/kaggle/Titanic/all/train.csv" # Path to data file
data = pd.read_csv(datafile, sep = ",")
datafile_test = "/home/crohr/kaggle/Titanic/all/test.csv" # Path to data file
data_test = pd.read_csv(datafile_test, sep = ",")
data_test = data_test.set_index("PassengerId")

#data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data

# ----------------------------------------------------------------------------
# Reformat columns (if needed)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Data inspection
# ----------------------------------------------------------------------------
# Inspect the data
data.shape  # Shape
data.dtypes # Data type in each column
data.head() # Inspect first rows

# Describe the data
data_stats = data.describe()

# ----------------------------------------------------------------------------
# Find correlations
# ----------------------------------------------------------------------------
plt = correlation_heatmap(data)

# Remove columns, where, the presence of 0 makes sense
data = data.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis = 1)
data_test = data_test.drop(["Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis = 1)



X_train = data.ix[:,1:5]
X_test = data_test.ix[:,0:4]
y_train = data.ix[:,0]


feature_scale = 1
# Feature scaling
if feature_scale == 1:
    X_train, X_test = scale_data(X_train, X_test)

# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------
# Choose the algorithms    
#    LogisticRegression
#    KNeighbors
#    GaussianNB
#    SVC
#    LinearSVC
#    RandomForestClassifier
#    DecisionTreeRegressor
models_probe = [
#        'LogisticRegression', 
#        'KNeighbors',
#        'GaussianNB', 
        'SVC', 
        #'LinearSVC',
        'RandomForestClassifier', 
        # 'DecisionTreeRegressor'
        ]

# Comment unnecesary steps
analysis_steps = [
         'model_selection',
         'cross_validation',
         'hyperparameters_tunning'
        ]

model = perform_analysis(models_probe, analysis_steps, X_train, y_train, X_test, y_test)
y_pred = model.predict(X_test)

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, y_pred
print(classification_report(y_true, y_pred))
print()

