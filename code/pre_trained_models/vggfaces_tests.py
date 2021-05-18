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

###########test_activations_vgg_face
model = VGGFace(model = 'vgg16', weights = 'vggface')
img_path = '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/small_test/images/AF1.jpg'
img = load_img(img_path, target_size=(224, 224))
image = img_to_array(img)
img = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # ?
image = preprocess_input(img)
activations = keract.get_activations(model, image)

with open('/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/small_test/activations_vggface.txt',"w") as file:
        file.write(str(activations))


###########test_activations_imagenet
model = VGG16(weights = 'imagenet')
img_path = '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/small_test/images/AF1.jpg'
img = load_img(img_path, target_size=(224, 224))
image = img_to_array(img)
img = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # ?
image = preprocess_input(img)
activations = keract.get_activations(model, image)

with open('/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/small_test/activations_imagenet.txt',"w") as file:
        file.write(str(activations))