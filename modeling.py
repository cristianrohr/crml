#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:00:33 2018

@author: crohr
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Import all the algorithms we want to test
from models import get_models

def cross_validation(classifier, X_train, y_train, folds = 10, message = True):
    """
    Perform k-fold Cross Validation using cross_val_score in sklearn.model_selection
    
    Parameteres:
        X_train: Training features
        y_train: Training labels
        folds:   k value
        message: Print statistics and metrics
        
    Return:
        List of accuracies for each fold
    """
    
    kfold = KFold(n_splits=folds, random_state=7)
    all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=kfold) 
    
    if message == True:
        print("Accuracies for {}-Fold CV= {}".format(folds, all_accuracies))
        print("The mean value is %f, or %f" % (all_accuracies.mean(), all_accuracies.mean()*100))
        print("The standar deviation value is %f, or %f" % (all_accuracies.std(), all_accuracies.std()*100))
    
    return all_accuracies


def grid_search(classifier, X_train, y_train, gparams, 
                score = 'accuracy', cv = 10, jobs = -1):
    """
    Perform a model hyperparameters grid search usin GridSearchCV in 
    sklearn.model_selection
    
    Parameteres:
        classifier: sklearn ML algorithm
        X_train:    Training features
        y_train:    Training labels
        gparams:    Dictionary of parameters to be screened
        score:      Scoring metric
        cv:         K value for Cross Validation
        jobs:       Cores (-1 for all)
        
    Return:
        Best model
    """
    # Initialize GridSearchCV class
    gd_sr = GridSearchCV(estimator=classifier,  
                     param_grid=gparams,
                     scoring=score,
                     cv=cv,
                     n_jobs=jobs)
    
    # Fit the trainning data
    gd_sr.fit(X_train, y_train)

    # Results
    best_result = gd_sr.best_score_
    print("Best result = {}".format(best_result))
    
    # Get the best parameters    
    best_parameters = gd_sr.best_params_  
    print("Best parameters = {}".format(best_parameters))
    
#    print("Grid scores on development set:")
#    print()
#    means = gd_sr.cv_results_['mean_test_score']
#    stds = gd_sr.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, gd_sr.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
    
    
    return gd_sr
    

def select_model(models_probe, X_train, y_train, cv = 10):
    """
    Compare performance of different models in a dataset
    Print metrics for each model, and a boxplot with each model performance
    
    Parameteres:
        models_probe: List of models to test
        X_train:      Training features
        y_train:      Training labels
        cv:           K value for Cross Validation
        
    Return:
        None
    """
    
    # Get the models implementations
    models = get_models(models_probe)

    # Variables to store results  
    results = []
    names = []
   
    # Every algorithm is tested and results are
    # collected and printed
    for name, model in models:
        cv_results = cross_validation(model, X_train, y_train, folds=cv, message = False)
        results.append(cv_results)
        names.append(name)
        msg = "Model Selection - %s: %f (%f)" % (
            name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    
    # Check the best performing model
    results_np = np.array(results)
    results_mean = np.mean(results_np, axis = 1)
    best_model = names[np.argmax(results_mean)]
                     
    return best_model