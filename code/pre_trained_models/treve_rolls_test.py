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
#####################################################################################
#SETTINGS:
#####################################################################################
vec = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,26,28,29,30,31,32,33,34,35,36,37,38,39,40]

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
        denominator += (each**2)/length 
    tr=1 - (((numerator/length)**2) /denominator)
    return tr 
#####################################################################################
def treves_rolls2(vector):
    """
    compute modified treve-rolls population sparsennes, formula from (wilmore et all, 2000)
    """    
    length = len(vector)
    numerator = functools.reduce(lambda x,y: abs(x) + abs(y), vector)
    denominator = functools.reduce(lambda x,y: (x**2)/length + (y**2)/length, vector)
    tr=1 - (((numerator/length)**2) /denominator)
    return tr 
#####################################################################################
def treves_rolls3(vector):
    """
    compute modified treve-rolls population sparsennes, formula from (wilmore et all, 2000)
    """    
    length = len(vector)
    numerator = functools.reduce(lambda x,y: abs(x) + abs(y), vector)
    denominator = functools.reduce(lambda x,y: (x**2)/length + (y**2)/length, vector)
    print(denominator)
    tr=1 - (((numerator/length)**2) /denominator)
    return tr 
#####################################################################################
#CODE
#####################################################################################
import time
t1 = time.time()
print('tr1: ',treves_rolls(vec))
t2 = time.time()
print('tr2: ',treves_rolls2(vec))
t3 = time.time()
print('time_tr1: ',t2 - t1)
print('time_tr2: ',t3 - t2)