#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#[EN]The metrics can be computed in several ways on the layers, these different functions prtmettent to apply these different ways
#A mathematical formalization, with formulas, will be produced to describe them formally

#[FR]Les métriques peuvent être calculées de plusieurs manières sur les couches, ces différentes fonctions prtmettent d'appliquer ces différentes façons de faire
# Une formalisation mathématique, avec des formules, sera produite pour les décrire formellement

#1. compute_filter: Compute chosen formula (gini, treve-rolls, l1 norm...) for each filter (flatten), and compute the mean of them

#2. compute_flatten: Create a flatten vector from each layer and compute chosen formula (gini, treve-rolls, l1 norm...) on it"

#3. compute_channel: Compute chosen formula (gini, treve-rolls, l1 norm...) for each channel (1D vector on z dimension), and compute the mean of them

#4. compute_activations: Executes one of the 3 previous functions according to the approach passed in parameter

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
from numpy.linalg import norm
import statistics as st
import scipy
import sys
#personnal librairies
sys.path.insert(1,'../../code/functions')
import sparsenesslib.metrics as metrics
#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def compute_filter(activations, activations_dict,layer,formula,k):
    '''
    Compute chosen formula (gini, treve-rolls, l1 norm...) for each filter (flatten), and compute the mean of them
    '''    
    filter = []

    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)

    tuple = activations[layer].shape
    liste = list(tuple)    
    nb_channels = liste[3] 

    for each in range(0, nb_channels-1): #on itère sur chaque filtre (profondeur)
        filter.append([])
        index_row = 0
        for row in activations[layer][0]:
            index_column = 0
            for column in activations[layer][0][index_row]:            
                filter[each].append(activations[layer][0][index_row][index_column][each])                 
                index_column += 1
            index_row += 1

    filter_metrics = []

    for each in filter:
        if formula == 'L0':                
            filter_metrics.append(1 - (norm(each, 0)/len(each)))
        if formula == 'L1':                
            filter_metrics.append(norm(each, 1))
        elif formula == 'treve-rolls':
            filter_metrics.append(metrics.treves_rolls(each))
        elif formula == 'gini':            
            filter_metrics.append(metrics.gini(each))
        elif formula == 'kurtosis':
            filter_metrics.append(scipy.stats.kurtosis(each))
    activations_dict[layer] = st.mean(filter_metrics)

####################################################/length#################################
def compute_flatten(activations, activations_dict,layer,formula,k):
    '''
    Create a flatten vector from each layer and compute chosen formula (gini, treve-rolls, l1 norm...) on it"
    '''
    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)
    
    arr = activations[layer].flatten()
    if formula == 'L0':    
        activations_dict[layer] = (1 - (norm(arr, 0)/len(arr)))
    if formula == 'L1':    
        activations_dict[layer] = (norm(arr, 1))        
    elif formula == 'treve-rolls':        
        activations_dict[layer] = (metrics.treves_rolls(arr))
    elif formula == 'gini':
        activations_dict[layer] = (metrics.gini(arr))
    elif formula == 'kurtosis':
        activations_dict[layer] = (scipy.stats.kurtosis(arr))
#####################################################################################
def compute_channel(activations, activations_dict,layer,formula,k):
    '''
    Compute chosen formula (gini, treve-rolls, l1 norm...) for each channel, and compute the mean of them
    '''
    channels = []
    index_row = 0

    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)

    for row in activations[layer][0]:
        index_column = 0
        for column in activations[layer][0][index_row]:            
            channel = activations[layer][0][index_row][index_column]      
            if formula == 'L0':                
                channels.append(1-(norm(channel, 0)/len(channel)))                
            elif formula == 'L1':                
                channels.append(norm(channel, 1))
            elif formula == 'treve-rolls':
                channels.append(metrics.treves_rolls(channel))
            elif formula == 'gini':
                channels.append(metrics.gini(channel))
            elif formula == 'kurtosis':
                channels.append(scipy.stats.kurtosis(channel))
            index_column += 1
        index_row += 1    
    activations_dict[layer] = st.mean(channels)
#####################################################################################
def compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula, k):
    '''
    executes one of the 3 previous functions according to the approach passed in parameter
    '''
        
    for layer in layers:            
        if computation == 'channel':                
            if layer in flatten_layers:
                compute_flatten(activations, activations_dict, layer,formula,k)       
            else:                     
                compute_channel(activations, activations_dict, layer, formula,k)
        elif computation == 'filter':
            if layer in flatten_layers:
                compute_flatten(activations, activations_dict, layer,formula,k)       
            else:                     
                compute_filter(activations, activations_dict, layer, formula,k)
        elif computation == 'flatten':
            compute_flatten(activations, activations_dict, layer, formula,k)            
                
        else: print('ERROR: computation setting isnt channel or flatten')