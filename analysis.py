#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 00:02:02 2018

@author: crohr
"""

from modeling import cross_validation
from modeling import grid_search
from modeling import select_model
from models import get_models
from models import get_hyperparameters

def perform_analysis(models_probe, steps, X_train, y_train, X_test = None, y_test = None):
    if 'model_selection' in steps:
                
        print("");print("-----------------------------------------------")
        print("Model selection")
        print("-----------------------------------------------");print("")
        
        best_model = select_model(models_probe, X_train, y_train)
    
        print("Best model is: " + best_model)
        print("");print("")
                
    if 'cross_validation' in steps:
        
        print("");print("-----------------------------------------------")
        print("Cross Validation")
        print("-----------------------------------------------");print("")
       
        # Reevaluate best performing algorithm, to check process sanity, or
        # the selected model
        try:
            models_probe = [best_model]
            print("Model CV is: " + best_model)
            model = get_models(models_probe)
            
        except NameError:
            model_use = ""
            if len(models_probe) > 1:
                model_use = models_probe[0]
            else:
                model_use = models_probe
            print("Model CV is: " + model_use)
            model = get_models(model_use)
            
        result = cross_validation(model[0][1], X_train, y_train, message = True)
    
    if 'hyperparameters_tunning' in steps:  
        print("");print("-----------------------------------------------")
        print("Hyperparameters tunning")
        print("-----------------------------------------------");print("")

        # Grid Search
        try:
            models_probe = [best_model]
            model = get_models(models_probe)
            hyperparams = get_hyperparameters(models_probe)
        except NameError:
            model_use = ""
            if len(models_probe) > 1:
                model_use = models_probe[0]
            else:
                model_use = models_probe
            model = get_models(model_use)
            hyperparams = get_hyperparameters(model_use)            
            

        result = grid_search(model[0][1], X_train, y_train, hyperparams[0][1])
                
        return result
        