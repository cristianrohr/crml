#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:20:18 2018

@author: crohr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_features_labels(data, label_col = -1, impute = False, mv = 'NaN', \
                        strategy = 'mean'):
    """
    Get the training features and the label for the data
    * Optionally impute missing data using the sklearn.preprocessing Imputer module
    
    Parameters:
        data:       data
        label_col:  column index with labels (default last column)
        impute:     impute missing data (boolean) 
        mv:         missing_values encoding for imputation
        strategy:   imputation strategy
        
    Return:
        X, y Features and labels
    """
    
    X = data.iloc[:, 0:label_col].values
    y = data.iloc[:,label_col].values
    
    if impute:
        imputer = Imputer(missing_values = mv, strategy = strategy)
        X = imputer.fit_transform(X)
    
    return [X, y]


def get_training_testing(X, y, test_size = 0):
    """
    Splits the dataset into training and testing data
    
    Parameters:
        X:         Features
        y:         Labels
        test_size: proportion of samples for validation set
        
    Return:
        X_train, X_test, y_train, y_test features and labels for 
        training and testing
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 7)
    
    return (X_train, X_test, y_train, y_test)
        

def scale_data(X_train, X_test, scaler = "Standar"):
    """
    Data normalization with feature scaling
    
    Parameteres:
        X_train:   Training features
        X_test:    Testing features
        scaler:    Scaling algorithm
        
    Return:
        train_features, test_features scaled features for training and testing
    """
    
    if scaler == "Standar":
        feature_scaler = StandardScaler()
        train_features = feature_scaler.fit_transform(X_train)
        test_features = feature_scaler.transform(X_test)
    elif scaler == "MinMax":
        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(X_train)
        test_features = scaler.transform(X_test)
        
    return [train_features, test_features]
    

def correlation_heatmap(data):
    # correlation heatmap 
    corr = data.corr()
    plt.figure(figsize = (9, 7))
    sns.heatmap(corr, cmap="RdBu",
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show
    return plt


def encode_categorical(X_train, X_test, column):
    # Limit to categorial data using select_dtypes()
    X = X_train.select_dtypes(include=[object])
    # Encoding categorical data, with value between 0 and n-1 classes
    labelencoder = LabelEncoder()
    X_2 = X.apply(labelencoder.fit_transform)
    
    # Fit one hot encoder
    onehotencoder = OneHotEncoder()
    onehotencoder.fit(X_2)
    
    # Transform
    onehotlabels = onehotencoder.transform(X_2).toarray()
    X_train = onehotencoder.fit_transform(X_train.ix[:, column])
    return X