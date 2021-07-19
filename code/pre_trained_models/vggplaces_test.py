#!/usr/bin/env python3
#####################################################################################
# LIBRAIRIES:
#####################################################################################
import time
import os
from tensorflow.keras.preprocessing.image import load_img
import csv
import keract  # low to import
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from numpy.linalg import norm
import pandas
import statistics as st
import scipy
import sys
sys.path.insert(1,'/home/sonia/Bureau/internship_cefe_2021/code/pre_trained_models')
import vggplaces.vgg16_places_365 as places


from tensorflow.keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from vggplaces import vgg16_places_365
from scipy.stats import linregress

import statistics as st


model = places.VGG16_Places365(weights='places')
print (model.summary.layer())