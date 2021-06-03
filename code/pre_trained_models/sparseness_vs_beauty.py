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
import sparsenesslib as spl #personnal library
import statistics as st
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             
#bdd = 'BIGTEST' #'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
list_bdd = ['CFD'] #"['CFD','MART','JEN','SCUT-FBP']"
model_name = 'VGG16'  # 'vgg16, resnet (...)'
#weights = 'vggface' #'imagenet','vggface'
list_weights = ['vggface'] #['vggface','imagenet']
list_metrics = ['L0'] #['L1','gini_flatten','gini_channel','gini_filter','kurtosis']
computer = 'sonia'
freqmod = 20 #frequency of prints, if 5: print for 1/5 images
#####################################################################################
#CODE
#####################################################################################
k = 1
for bdd in list_bdd:    
    for weight in list_weights:
        for metric in list_metrics:
            print('############################################################_STEP: ',k,'/40 ',bdd,', ',weight,', ',metric)
            spl.layers_analysis(bdd,weight,metric, model_name, computer, freqmod,k)
            k += 1


