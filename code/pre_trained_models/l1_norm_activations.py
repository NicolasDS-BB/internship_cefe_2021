#!/usr/bin/env python
#####################################################################################
# LIBRAIRIES:
#####################################################################################
import time
t0 = time.time()
import os
import csv
from io import BytesIO
from PIL import Image
import keract  # low to import
import sys
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras_vggface.vggface import VGGFace
from numpy.linalg import norm
import pandas
import statistics as st
import functools
from scipy.stats import linregress
#####################################################################################
#SETTINGS:
#####################################################################################
bdd = 'MART' #'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
model_name = 'VGG16'  # 'vgg16, resnet (...)'
weights = 'imagenet' #'imagenet','vggface'
#####################################################################################
if bdd == 'CFD':
    labels_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/labels_CFD.csv'
    images_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/images'
    log_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/log_correlations_CFD'
elif bdd == 'JEN':
    labels_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/labels_JEN.csv'
    images_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/images'
    log_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/log_correlations_JEN'
elif bdd == 'SCUT-FBP':
    labels_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
    images_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/SCUT-FBP/images'
    log_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/SCUT-FBP/log_correlations_SCUT-FBP'
elif bdd == 'MART':
    labels_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/MART/labels_MART.csv'
    images_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/MART/images'
    log_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/MART/log_correlations_MART'
elif bdd == 'SMALLTEST':
    labels_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/small_test/labels_test.csv'
    images_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/small_test/images'
    log_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/small_test/log_correlations_test'
elif bdd == 'BIGTEST':
    labels_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/big_test/labels_bigtest.csv'
    images_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/big_test/images'
    log_path ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/big_test/log_correlations_bigtest'
#####################################################################################
if model_name == 'VGG16':
    if weights == 'imagenet':
        model = VGG16(weights = 'imagenet')
        layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
        'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
        'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2'] 
        flatten_layers = ['fc1','fc2','flatten']
    elif weights == 'vggface':
        model = VGGFace(model = 'vgg16', weights = 'vggface')
        layers = ['input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3',
        'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','flatten',
        'fc6/relu','fc7/relu']
        """layers = ['input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3',
        'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','flatten','fc6',
        'fc6/relu','fc7','fc7/relu','fc8','fc8/softmax']""" 
        flatten_layers = ['flatten','fc6','fc6/relu','fc7','fc7/relu','fc8','fc8/softmax']
elif model_name == 'resnet50':
    if weights == 'imagenet': 
        print('error, model not configured')
    elif weights == 'vggfaces':
        print('error, model not configured')
#####################################################################################
# VARIABLES:
#####################################################################################
dict_flatten_norms = {}
dict_channel_norms = {}
dict_channel_TR = {}
dict_flatten_TR = {}
dict_labels = {}
#####################################################################################
# PROCEDURE:
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
    arr = activations[layer].flatten()
    if formula == 'L1':    
        activations_dict[layer] = (norm(arr, 1))        
    elif formula == 'treve-rolls':        
        activations_dict[layer] = (treves_rolls(arr))
#####################################################################################
def compute_channel(activations, activations_dict,layer,formula):
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
def compute_norm_activations(path, dict_output, layers, computation, formula):
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

        for layer in layers:            
            if computation == 'channel':                
                if layer in flatten_layers:
                    compute_flatten(activations, activations_dict, layer,formula)       
                else:                     
                    compute_channel(activations, activations_dict, layer, formula)
            elif computation == 'flatten':
                compute_flatten(activations, activations_dict, layer, formula)
                
            else: print('ERROR: computation setting isnt channel or flatten')

        dict_output[each] = activations_dict
#####################################################################################
# CODE:
#####################################################################################
#compute_norm_activations(images_path,dict_channel_norms, layers, 'channel', 'L1')
compute_norm_activations(images_path, dict_flatten_norms, layers, 'flatten', 'L1')
#compute_norm_activations(images_path,dict_channel_TR, layers, 'channel', 'treve-rolls') #extrèmement lent même sur le test
compute_norm_activations(images_path, dict_flatten_TR, layers, 'flatten', 'treve-rolls')
#####################################################################################
# PROCEDURE:
#####################################################################################
# on importe les notes de beauté/attractivité dans un dict {image.jpg:score}
def parse_rates(labels_path , dict_output):
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
# CODE:
#####################################################################################
parse_rates(labels_path, dict_labels)
#####################################################################################
# FUNCTION:
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
#####################################################################################
# CODE:
#####################################################################################
df_flatten_norms = create_dataframe(dict_labels, dict_flatten_norms)
df_flatten_TR = create_dataframe(dict_labels, dict_flatten_TR)
#df_channel_norms = create_dataframe(dict_labels,dict_channel_norms)
#df_channel_TR = create_dataframe(dict_labels,dict_channel_TR)

with open(log_path +'_'+ weights+'_L1_TR.csv',"w") as file:
         
    file.write('layer')
    file.write(';')
    file.write('R_flatten_L1_norm')
    file.write(';')
    file.write('pvalue_L1_norm')
    file.write(';')
    file.write('R_flatten_treve-rolls')
    file.write(';')
    file.write('pvalue_treve-rolls')
    file.write(';')
    file.write('R_L1_vs_treve-rolls')
    file.write(';')
    file.write('pvalue_L1_vs_treve-rolls')
    file.write(';')
    file.write('\n') 
    for layer in layers:
        file.write(layer)
        file.write(';')
        #flatten_L1
        list_flatten_norms = list(df_flatten_norms[layer])
        list_flatten_rate = list(df_flatten_norms['rate'])
        reg = linregress(list_flatten_norms,list_flatten_rate)       
        coeff = str(reg.rvalue)
        pvalue = str(reg.pvalue)        
        file.write(coeff)
        file.write(';')
        file.write(pvalue)
        file.write(';')        
        #flatten_TR
        list_flatten_TR = list(df_flatten_TR[layer])        
        reg = linregress(list_flatten_TR,list_flatten_rate)       
        coeff = str(reg.rvalue)
        pvalue = str(reg.pvalue)        
        file.write(coeff)
        file.write(';')
        file.write(pvalue)
        file.write(';') 
        #corr_TR_vs_L1        
        reg = linregress(list_flatten_TR,list_flatten_norms)       
        coeff = str(reg.rvalue)
        pvalue = str(reg.pvalue)        
        file.write(coeff)
        file.write(';')
        file.write(pvalue)
        file.write(';') 
        file.write("\n")
    
    t1 = time.time()
    file.write("\n")
    file.write("##############")
    file.write("\n")
    total = (t1-t0)/60
    total = str(total)
    file.write("time : ")    
    file.write(total)
    file.write(' minutes')



