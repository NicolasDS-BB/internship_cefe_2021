# gini(vector)
# treve_rolls(vector)
# compute_flatten(activations, activations_dict,layer,formula)
# def compute_channel(activations, activations_dict,layer,formula)
# compute_norm_activations(model, flatten_layers, path, dict_output, layers, computation, formula)
# def parse_rates(labels_path , dict_labels)
# def create_dataframe(dict_rates, dict_norm)


#####################################################################################
# LIBRAIRIES:
#####################################################################################
import time
t0 = time.time()
import os
import csv
import keract  # low to import
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy.linalg import norm
import pandas
import statistics as st
#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def gini(vector):
    """Calculate the Gini coefficient of a numpy array."""
    
    if np.amin(vector) < 0:
        # Values cannot be negative:
        vector -= np.amin(vector)
    # Values cannot be 0:
    vector += 0.0000001
    # Values must be sorted:
    vector = np.sort(vector)
    # Index per array element:
    index = np.arange(1,vector.shape[0]+1)
    # Number of array elements:
    n = vector.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * vector)) / (n * np.sum(vector)))
#####################################################################################
def treves_rolls(vector):
    """
    compute modified treve-rolls population sparsennes, formula from (wilmore et all, 2000)
    """
    denominator = 0
    numerator = 0
    length = len(vector)
    for each in vector:
        numerator += abs(each)
        denominator += (each*each)/length 
    tr=1 - (((numerator/length)*(numerator/length)) /denominator)
    return tr 
####################################################/length#################################
def compute_flatten(activations, activations_dict,layer,formula):
    '''
    Create a flatten vector from each layer and compute chosen formula (gini, treve-rolls, l1 norm...) on it"
    '''
    arr = activations[layer].flatten()
    if formula == 'L1':    
        activations_dict[layer] = (norm(arr, 1))        
    elif formula == 'treve-rolls':        
        activations_dict[layer] = (treves_rolls(arr))
    elif formula == 'gini':
        activations_dict[layer] = (gini(arr))
#####################################################################################
def compute_channel(activations, activations_dict,layer,formula):
    '''
    Compute chosen formula (gini, treve-rolls, l1 norm...)
    '''
    channels = []
    index_row = 0
    for row in activations[layer][0]:
        index_column = 0
        '''print('dimlayer ',layer,': ',activations[layer][0].shape)
        print('dimlayerbis ',layer,': ',activations[layer].shape)'''
        for column in activations[layer][0][index_row]:            
            channel = activations[layer][0][index_row][index_column] 
            #print('dimchannel:',channel.shape)
            #print('###############')            
            if formula == 'L1':                
                channels.append(norm(channel, 1))
            elif formula == 'treve-rolls':
                channels.append(treves_rolls(channel))
            index_column += 1
        index_row += 1    
    activations_dict[layer] = st.mean(channels)
#####################################################################################
def compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula):
        
        for layer in layers:            
            if computation == 'channel':                
                if layer in flatten_layers:
                    compute_flatten(activations, activations_dict, layer,formula)       
                else:                     
                    compute_channel(activations, activations_dict, layer, formula)
            elif computation == 'flatten':
                compute_flatten(activations, activations_dict, layer, formula)            
                
            else: print('ERROR: computation setting isnt channel or flatten')
#####################################################################################
def compute_sparseness_metrics_activations(model, flatten_layers, path, dict_output, layers, computation, formula):
    """
    compute the l1 norm of the layers given in the list *layers*
    of the images contained in the directory *path*
    by one of those 2 modes: flatten or channel (cf previous functions)
    and store them in the dictionary *dict_output*.
    """
    imgs = [f for f in os.listdir(path)]    
    i = 1
    for each in imgs:
        print('###### picture n°',i,'/',len(imgs),'for ',formula)
        i += 1
        img_path = path + "/" + each
        img = load_img(img_path, target_size=(224, 224))
        image = img_to_array(img)
        img = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))  # ?
        image = preprocess_input(img)
        # récupération des activations
        activations = keract.get_activations(model, image)
        activations_dict = {}
        compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula)
        dict_output[each] = activations_dict
#####################################################################################
def parse_rates(labels_path , dict_labels):
    """
    Stores notes and image names contained in *path* 
    in *dict* as {name:note}    
    """
    with open(labels_path, newline='') as labels:
        reader = csv.reader(labels)
        for line in reader:
            key = line[0]
            rate = line[1]
            dict_labels[key] = float(rate)
#####################################################################################
def create_dataframe(dict_rates, dict_norm):
    """
    Creates a pandas dataframe that has a beauty score associates
    the L1 of the associated image layers
    rows: images, column 1: notes, column 2 to n: L1 norms
    """
    df1 = pandas.DataFrame.from_dict(dict_rates, orient='index', columns = ['rate'])
    df2 = pandas.DataFrame.from_dict(dict_norm, orient='index')     

    df = pandas.concat([df1, df2], axis = 1)     
    return df