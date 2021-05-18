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
import sparsenesslib as spl #personnal library
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             
bdd = 'SMALLTEST' #'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
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
# CODE:
#####################################################################################
#spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_channel_norms, layers, 'channel', 'L1')
spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path, dict_flatten_norms, layers, 'flatten', 'gini')
#spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_channel_TR, layers, 'channel', 'treve-rolls') #extrèmement lent même sur le test
#spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path, dict_flatten_TR, layers, 'flatten', 'treve-rolls')

spl.parse_rates(labels_path, dict_labels)

df_flatten_norms = spl.create_dataframe(dict_labels, dict_flatten_norms)
#df_flatten_TR = create_dataframe(dict_labels, dict_flatten_TR)
#df_channel_norms = create_dataframe(dict_labels,dict_channel_norms)
#df_channel_TR = create_dataframe(dict_labels,dict_channel_TR)

with open(log_path +'_'+ weights+'_gini.csv',"w") as file:
         
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
        '''list_flatten_TR = list(df_flatten_TR[layer])        
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
        file.write(';') '''
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



