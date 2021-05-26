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
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             
bdd = 'BIGTEST' #'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
model_name = 'VGG16'  # 'vgg16, resnet (...)'
weights = 'vggface' #'imagenet','vggface'
computer = 'sonia'
freqmod = 5 #frequency of prints, if 5: print for 1/5 images
#####################################################################################
if computer == 'sonia': #databases aren't in repo bc they need to be in DATA partition of the pc (more space)
    if bdd == 'CFD':
        labels_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/labels_CFD.csv'
        images_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/images'
        log_path ='../../results/CFD/log_CFD'
    elif bdd == 'JEN':
        labels_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/labels_JEN.csv'
        images_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/images'
        log_path ='../../results/JEN/log_JEN'
    elif bdd == 'SCUT-FBP':
        labels_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
        images_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/images'
        log_path ='../../results/SCUT-FBP/log_SCUT-FBP'
    elif bdd == 'MART':
        labels_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/labels_MART.csv'
        images_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/images'
        log_path ='../../results/MART/log_MART'
    elif bdd == 'SMALLTEST':
        labels_path ='../../data/redesigned/small_test/labels_test.csv'
        images_path ='../../data/redesigned/small_test/images'
        log_path ='../../results/smalltest/log_test'
    elif bdd == 'BIGTEST':
        
        labels_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/labels_bigtest.csv'
        images_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/images'
        log_path ='../../results/bigtest/log_bigtest'

else: #all paths are relative paths
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
#####################################################################################
# VARIABLES:
#####################################################################################
dict_flatten_norms = {}
dict_channel_norms = {}
dict_filter_norms = {}

dict_channel_tr = {}
dict_flatten_tr = {}
dict_filter_tr = {}

dict_channel_gini = {}
dict_flatten_gini = {}
dict_filter_gini = {}

dict_channel_kurt = {}
dict_flatten_kurt = {}
dict_filter_kurt = {}

dict_labels = {}
#####################################################################################
# CODE:
#####################################################################################
spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_flatten_norms, layers, 'flatten', 'L1', freqmod)
spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path, dict_flatten_tr, layers, 'flatten', 'treve-rolls', freqmod)
spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_flatten_gini, layers, 'flatten', 'gini', freqmod) 
spl.compute_sparseness_metrics_activations(model,flatten_layers, images_path, dict_flatten_kurt, layers, 'flatten', 'kurtosis', freqmod)

spl.parse_rates(labels_path, dict_labels)

df_flatten_norms = spl.create_dataframe(dict_labels, dict_flatten_norms)
df_flatten_tr = spl.create_dataframe(dict_labels, dict_flatten_tr)
df_flatten_gini = spl.create_dataframe(dict_labels,dict_flatten_gini)
df_flatten_kurt = spl.create_dataframe(dict_labels,dict_flatten_kurt)


with open(log_path +'_'+ weights+'_allcorr.csv',"w") as file:
         
    file.write('layer')
    file.write(';')
    file.write('corr_L1_vs_TR')
    file.write(';')
    file.write('corr_L1_vs_Gini')
    file.write(';')
    file.write('corr_L1_vs_Kurtosis')
    file.write(';')
    file.write('corr_TR_vs_Gini')
    file.write(';')
    file.write('corr_TR_vs_Kurtosis')
    file.write(';')
    file.write('corr_Gini_vs_Kurtosis')
    file.write(';')
    file.write('\n') 
    for layer in layers:
        file.write(layer)
        file.write(';')
        #corr_L1_vs_TR
        l1 = list(df_flatten_norms[layer])
        l2 = list(df_flatten_tr[layer])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        file.write(coeff)
        file.write(';')    
        #corr_L1_vs_Gini
        l1 = list(df_flatten_norms[layer])
        l2 = list(df_flatten_gini[layer])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        file.write(coeff)
        file.write(';') 
        #corr_L1_vs_Kurtosis
        l1 = list(df_flatten_norms[layer])
        l2 = list(df_flatten_kurt[layer])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        file.write(coeff)
        file.write(';') 
        #corr_TR_vs_Gini
        l1 = list(df_flatten_tr[layer])
        l2 = list(df_flatten_gini[layer])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        file.write(coeff)
        file.write(';') 
        #corr_TR_vs_Kurtosis
        l1 = list(df_flatten_tr[layer])
        l2 = list(df_flatten_kurt[layer])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        file.write(coeff)
        file.write(';') 
        #corr_Gini_vs_Kurtosis
        l1 = list(df_flatten_gini[layer])
        l2 = list(df_flatten_kurt[layer])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        file.write(coeff)
        file.write(';') 
        '''#flatten_L1
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
        list_filter_norms = list(df_filter_norms[layer])        
        reg = linregress(list_filter_norms,list_flatten_rate)       
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



