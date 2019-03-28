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
datafile = "datasets/train_sklearn.csv"  # Path to data file
testfile = "datasets/test.csv"  # Path to data file
data = pd.read_csv(datafile, decimal=".", na_values="?", sep=",", comment='@')
test = pd.read_csv(testfile, decimal=".", na_values="?", sep=",", comment='@')


# FIltro las columnas de test
variables_train = list(data.columns.values)
variables_train = [x for x in variables_train if x != "C"]
test = test[variables_train]

# ----------------------------------------------------------------------------
# Reformat columns (if needed)
# ----------------------------------------------------------------------------
# data.columns = [
#    "NumTimesPrg", "PlGlcConc", "BloodP",
#    "SkinThick", "TwoHourSerIns", "BMI",
#    "DiPedFunc", "Age", "HasDiabetes"]

# data.columns = ["X1",
#    "X2", "X3", "X4",
#    "X5", "X6", "X7",
#    "X8", "X9", "Y"]

# ----------------------------------------------------------------------------
# Data inspection
# ----------------------------------------------------------------------------
# Inspect the data
data.shape  # Shape
data.dtypes  # Data type in each column
data.head()  # Inspect first rows

# Describe the data
data_stats = data.describe()

# ----------------------------------------------------------------------------
# Find correlations
# ----------------------------------------------------------------------------
plt = correlation_heatmap(data)

# ----------------------------------------------------------------------------
# Missing data analysis
# ----------------------------------------------------------------------------
# Get columns with 0 in data or '?'
# Iterate over min value in each column, and keep colnames
# cols_NA = [data_stats.columns[i] for i in range(len(data_stats.columns)) \
#          if data_stats.loc['min', data_stats.columns[i]] == 0.0 \
#        and i != (len(data_stats.columns) - 1)] # Not consider target label

# Count missing data per column
# print((data[cols_NA] == 0).sum())

# Remove columns, where, the presence of 0 makes sense
# cols_NA.remove("NumTimesPrg")

# mark zero values as missing or NaN
# data[cols_NA] = data[cols_NA].replace(0, np.NaN)

# count the number of NaN values in each column
print(data.isnull().sum())
print(data.head)


# ----------------------------------------------------------------------------
# Get features and labels
# ----------------------------------------------------------------------------
# Set parameters
impute_missing = 0  # Imputation is optional
feature_scale = 1  # Feature scaling is optional

# if impute_missing == 1:
# Impute missing data
#    [X, y] = get_features_labels(data, label_col = 51, impute = True, mv = "NaN", strategy = "mean")
# else:
# Get the features and labels
#   [X, y] = get_features_labels(data, label_col = 9)

# Split into training and testing datasets
#X_train, X_test, y_train, y_test = get_training_testing(X, y, test_size = 0.3)

X_train = data.loc[:, data.columns != 'C']
X_test = test
y_train = data["C"]

# Feature scaling
if feature_scale == 1:
    X_train, X_test = scale_data(X_train, X_test)

# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------
# Choose the algorithms

models_probe = [
    # -------------------
    # Clasificación:
    # -------------------
#    'LogisticRegression',
    'KNeighbors']
#    'LDA',
#    'QDA',
#    'GaussianNB',
#    'SVC',
#    'LinearSVC',
#    'RandomForestClassifier'


    # -------------------
    # Regresión:
    # -------------------
    # 'DecisionTreeRegressor'

# Comment unnecesary steps
analysis_steps = [
    'model_selection',
    'cross_validation',
    'hyperparameters_tunning'
]

#model = perform_analysis(models_probe, analysis_steps, X_train, y_train, X_test, y_test)
model = perform_analysis(models_probe, analysis_steps, X_train, y_train, X_test)
y_pred = model.predict(X_test)

# Guardo los resultados de knn
#d = {'Id':list(range(1, y_pred.size + 1)), 'Prediction': y_pred}
#df = pd.DataFrame(data=d)
#df.to_csv("KNN_sklearn_K19_p1_wdistance.csv", index = False)

#print("Detailed classification report:")
# print()
#print("The model is trained on the full development set.")
#print("The scores are computed on the full evaluation set.")
# print()
#y_true, y_pred = y_test, y_pred
#print(classification_report(y_true, y_pred))
# print()
