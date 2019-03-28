#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:14:28 2018

@author: crohr
"""

# Import all the algorithms we want
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# Regression
from sklearn.tree import DecisionTreeRegressor

import numpy as np

def get_models(models):
    
        # Dictionary with models
        models_dict = {'LogisticRegression': LogisticRegression(),
                       'KNeighbors': KNeighborsClassifier(),
                       'LDA': LinearDiscriminantAnalysis(),
                       'QDA': QuadraticDiscriminantAnalysis(),
                       'GaussianNB': GaussianNB(),
                       'SVC': SVC(),
                       'LinearSVC': LinearSVC(),
                       'RandomForestClassifier': RandomForestClassifier(),
                       'DecisionTreeRegressor': DecisionTreeRegressor()
        }
        
        models_to_use = [(val, models_dict[val]) for val in models]
        
        return models_to_use


def get_hyperparameters(models):    
    hyperparameters_dict = {
            
            'LogisticRegression': {'penalty': ['l1', 'l2'],
                                   'C': np.logspace(0, 4, 10)},
                      
            'KNeighbors': {'n_neighbors' : [19, 39],
                           'weights': ['uniform', 'distance'],
                           'p' : [1, 2],
                           #'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']},
                           'algorithm' : ['auto']},
                           
            'LDA': {},
            
            'QDA': {},
            
            'SVC': {'C': [0.1, 1, 10],
                   'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                   'class_weight': ['balanced'],
                   'gamma': [1, 0.1],
                   'degree': [1, 2, 3]},
                    
#            'SVC': {'C': [0.1, 1, 10, 100],
#                   'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#                   'class_weight': ['balanced', None],
#                   'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto'],
#                   'degree': [2, 3, 4]},
    
            'RandomForestClassifier': {'n_estimators': [1000, 5000],
                   'max_features': [5, 10, 15,  None],
                   'criterion': ['entropy', 'gini'],
                   'class_weight': ['balanced', "balanced_subsample"]}
    
    }
    
    hyperparameters = [(val, hyperparameters_dict[val]) for val in models]
        
    return hyperparameters


    
