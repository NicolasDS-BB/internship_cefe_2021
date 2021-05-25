#!/usr/bin/env python
#####################################################################################
# LIBRAIRIES:
#####################################################################################
import time
t0 = time.time()
from tensorflow.keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from scipy.stats import linregress
import PIL
import sys
sys.path.insert(1,'../../code/functions')
from sparsenesslib import *
import time
t0 = time.time()
import os
from tensorflow.keras.preprocessing.image import load_img
import csv
import keract  # low to import
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import seaborn
from numpy.linalg import norm
import pandas
import statistics as st
import matplotlib.pyplot as plt
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997 #useful for JEN bc images are heavy 
478940                             
bdd = 'SMALLTEST' #'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
model_name = 'VGG16'  # 'vgg16, resnet (...)'
weights = 'imagenet' #'imagenet','vggface'
computer = 'sonia'
#####################################################################################
if computer == 'sonia':
    if bdd == 'CFD':
        labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
        images_path ='../../data/redesigned/CFD/images'
        log_path ='../../data/redesigned/CFD/log_correlations_CFD'
    elif bdd == 'JEN':
        labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
        images_path ='../../data/redesigned/JEN/images'
        log_path ='../../data/redesigned/JEN/log_correlations_JEN'
    elif bdd == 'SCUT-FBP':
        labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
        images_path ='../../data/redesigned/SCUT-FBP/images'
        log_path ='../../data/redesigned/SCUT-FBP/log_correlations_SCUT-FBP'
    elif bdd == 'MART':
        labels_path ='../../data/redesigned/MART/labels_MART.csv'
        images_path ='../../data/redesigned/MART/images'
        log_path ='../../data/redesigned/MART/log_correlations_MART'
    elif bdd == 'SMALLTEST':
        labels_path ='../../data/redesigned/small_test/labels_test.csv'
        images_path ='../../data/redesigned/small_test/images'
        log_path ='../../data/redesigned/small_test/log_correlations_test'
    elif bdd == 'BIGTEST':
        labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
        images_path ='../../data/redesigned/big_test/images'
        log_path ='../../data/redesigned/big_test/log_correlations_bigtest'

elif computer == 'nicolas':
    if bdd == 'CFD':
        labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
        images_path ='../../data/redesigned/CFD/images'
        log_path ='../../data/redesigned/CFD/log_correlations_CFD'
    elif bdd == 'JEN':
        labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
        images_path ='../../data/redesigned/JEN/images'
        log_path ='../../data/redesigned/JEN/log_correlations_JEN'
    elif bdd == 'SCUT-FBP':
        labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
        images_path ='../../data/redesigned/SCUT-FBP/images'
        log_path ='../../data/redesigned/SCUT-FBP/log_correlations_SCUT-FBP'
    elif bdd == 'MART':
        labels_path ='../../data/redesigned/MART/labels_MART.csv'
        images_path ='../../data/redesigned/MART/images'
        log_path ='../../data/redesigned/MART/log_correlations_MART'
    elif bdd == 'SMALLTEST':
        labels_path ='../../data/redesigned/small_test/labels_test.csv'
        images_path ='../../data/redesigned/small_test/images'
        log_path ='../../data/redesigned/small_test/log_correlations_test'
    elif bdd == 'BIGTEST':
        labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
        images_path ='../../data/redesigned/big_test/images'
        log_path ='../../data/redesigned/big_test/log_correlations_bigtest'

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
####################################################################################
def compute_sparseness_metrics_activations(model,path,layers):
    """
    compute the l1 norm of the layers given in the list *layers*
    of the images contained in the directory *path*
    by one of those 2 modes: flatten or channel (cf previous functions)
    and store them in the dictionary *dict_output*.
    """
    imgs = [f for f in os.listdir(path)]    
    i = 0
    

    activations_dict = {}

    for each in imgs:       
        print('picture: ', i , '/', len(imgs)-1)         
        i += 1      
          
        img_path = path + "/" + each
        img = load_img(img_path, target_size=(224, 224))
        image = img_to_array(img)
        img = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))  # ?
        image = preprocess_input(img)
        # récupération des activations
        activations = keract.get_activations(model, image)
             
        
        for layer in layers:             
                       
            array = activations[layer].flatten()                                

            if layer in activations_dict:                  
                activations_dict[layer].append(array) 
            else:                 
                activations_dict[layer] = []
                activations_dict[layer].append(array)
    
    
    i = 1
    length = len(activations_dict)
    for each in activations_dict:
        
        print(i, "/",length)
        i += 1

        fusion = np.vstack(activations_dict[each])
        print('fusion done')

        plt.hist(fusion)
        plt.title(layer, fontsize=10)     
        plt.savefig(log_path +'_' + layer + 'graphe')


####################################################################################
compute_sparseness_metrics_activations(model,images_path,layers)

        