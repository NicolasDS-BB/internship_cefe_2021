#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#[EN]high level functions that organize the sequence of the other functions of the module, they would probably be difficult to reuse for another project

#[FR]fonctions de haut niveau qui organisent l'enchainement des autres fonctions du module, elles seraient probablement difficilement réutilisables pour un autre projet

#1. compute_sparseness_metrics_activations: compute metrics of the layers given in the list *layers* of the images contained in the directory *path*
    #by one of those 3 modes: flatten channel or filter (cf activations_structures subpackage) and store them in the dictionary *dict_output*.

#2. write_file: Writes the results of the performed analyses and their metadata in a structured csv file with 
    # a header line, 
    # results (one line per layer), 
    # a line with some '###', 
    # metadata

#3. layers_analysis: something like a main, but in a function (with all previous function),also, load paths, models/weights parameters and write log file

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import time
import os
from tensorflow.keras.preprocessing.image import load_img
import keract  
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from scipy.stats import linregress
import sys
import statistics as st
from datetime import date

#personnal librairies
sys.path.insert(1,'../../code/functions')
import sparsenesslib.metrics as metrics
import sparsenesslib.sparsenessmod as spm
import sparsenesslib.activations_structures as acst
#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def compute_sparseness_metrics_activations(model, flatten_layers, path, dict_output, layers, computation, formula, freqmod,k):
    '''
    compute metrics of the layers given in the list *layers*
    of the images contained in the directory *path*
    by one of those 3 modes: flatten channel or filter (cf activations_structures subpackage)
    and store them in the dictionary *dict_output*.
    '''
    imgs = [f for f in os.listdir(path)]    
    i = 1
    for each in imgs:

        if i%freqmod == 0:
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
        i += 1
        img_path = path + "/" + each
        img = load_img(img_path, target_size=(224, 224))
        image = img_to_array(img)
        img = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))  
        image = preprocess_input(img)
        # récupération des activations
        activations = keract.get_activations(model, image)
        activations_dict = {}
        acst.compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula,k)
        dict_output[each] = activations_dict
#####################################################################################
def write_file(log_path, bdd, weight, metric, df_metrics, df_reglog, df_scope, layers, k, t0):    
    '''
    Writes the results of the performed analyses and their metadata in a structured csv file with 
    - a header line, 
    - results (one line per layer), 
    - a line with some '###', 
    - metadata
    '''

    today = date.today()
    today = str(today)

    with open(log_path +'_'+bdd+'_'+weight+'_'+metric+'_'+today+'.csv',"w") as file:            
        #HEADER
        file.write('layer'+';'+'mean_'+str(metric)+';'+'sd_'+str(metric)+';'+'corr_beauty_VS_'+'metric'+';'+'pvalue'+';'+'\n')
        #VALUES for each layer
        for layer in layers:
            if layer[0:5] == 'input':
                layer = 'input' + '_' + str(k)
            file.write(layer+';')            
            #mean
            l1 = list(df_metrics[layer])
            file.write(str(st.mean(l1))+';')               
            #standard deviation
            l1 = list(df_metrics[layer])
            file.write(str(st.stdev(l1))+';')            
            #correlation with beauty
            l1 = list(df_metrics[layer])
            l2 = list(df_metrics['rate'])
            reg = linregress(l1,l2)
            coeff = str(reg.rvalue)         
            file.write(coeff +';')             
            #pvalue
            pvalue = str(reg.pvalue) 
            file.write(pvalue+';'+'\n')   
        
        #METADATA
        file.write('##############'+'\n')        
        file.write('bdd;'+ bdd + '\n')        
        file.write('weights;'+ weight + '\n')         
        file.write('metric;'+ metric + '\n')  
        t1 = time.time()      
        total = (t1-t0)/60
        total = str(total)
        file.write("time:;"+total+';minutes'+'\n')        
        file.write("date:;"+today+'\n')
        #correlation with scope
        l1 = list(df_scope['reglog'])        
        l2 = list(df_scope['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        file.write('coeff_scope: ;'+coeff+'\n') 
        #correlation with coeff of logistic regression
        l1 = list(df_reglog['reglog'])
        print('reglog:   ', l1)
        l2 = list(df_reglog['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        file.write('coeff_reglog: ;'+coeff+'\n')  
#####################################################################################
def layers_analysis(bdd,weight,metric, model_name, computer, freqmod,k = 1):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*'''

    t0 = time.time()

    if computer == 'sonia': #databases aren't in repo bc they need to be in DATA partition of the pc (more space)
        if bdd == 'CFD':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/labels_CFD.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/CFD/images'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/labels_JEN.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/JEN/images'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/SCUT-FBP/images'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/labels_MART.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/MART/images'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/labels_test.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/small_test/images'
            log_path ='../../results/smalltest/log_'
        elif bdd == 'BIGTEST':        
            labels_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/labels_bigtest.csv'
            images_path ='/media/sonia/DATA/data_nico/data/redesigned/big_test/images'
            log_path ='../../results/bigtest/log_'

    else: #for others configurations,all paths are relative paths in git repository
        if bdd == 'CFD':
            labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
            images_path ='../../data/redesigned/CFD/images'
            log_path ='../../results/CFD/log_'
        elif bdd == 'JEN':
            labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
            images_path ='../../data/redesigned/JEN/images'
            log_path ='../../results/JEN/log_'
        elif bdd == 'SCUT-FBP':
            labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
            images_path ='../../data/redesigned/SCUT-FBP/images'
            log_path ='../../results/SCUT-FBP/log_'
        elif bdd == 'MART':
            labels_path ='../../data/redesigned/MART/labels_MART.csv'
            images_path ='../../data/redesigned/MART/images'
            log_path ='../../results/MART/log_'
        elif bdd == 'SMALLTEST':                       
            labels_path ='../../data/redesigned/small_test/labels_test.csv'
            images_path ='../../data/redesigned/small_test/images'
            log_path ='../../results/smalltest/log_'            
        elif bdd == 'BIGTEST':
            labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
            images_path ='../../data/redesigned/big_test/images'
            log_path ='../../results/bigtest/log_'  

    if model_name == 'VGG16':
        if weight == 'imagenet':
            model = VGG16(weights = 'imagenet')
            layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2']
            flatten_layers = ['fc1','fc2','flatten']
        elif weight == 'vggface':
            model = VGGFace(model = 'vgg16', weights = 'vggface')
            layers = ['input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3',
            'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','flatten',
            'fc6/relu','fc7/relu']
            flatten_layers = ['flatten','fc6','fc6/relu','fc7','fc7/relu','fc8','fc8/softmax']
    elif model_name == 'resnet50':
        if weight == 'imagenet': 
            print('error, model not configured')
        elif weight == 'vggfaces':
            print('error, model not configured')  

    dict_compute_metric = {}
    dict_labels = {}

    if metric == 'L0':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'kurtosis':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'L1':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'gini_flatten':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', 'gini', freqmod, k)
    if metric == 'gini_channel':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'channel', 'gini', freqmod, k)
    if metric == 'gini_filter':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'filter', 'gini', freqmod, k)
    
    spm.parse_rates(labels_path, dict_labels)
    df_metrics = spm.create_dataframe(dict_labels, dict_compute_metric)    
    
    #écriture des histogrammes      
    metrics.histplot_metrics(layers, df_metrics, bdd, weight, metric, log_path,k)    
    #régression logistique    
    df_reglog = metrics.reglog(layers, df_metrics, dict_labels) 
    #minimum - maximum     
    df_scope = metrics.minmax(df_metrics,dict_labels)    
    #Gompertz    
    '''df_gompertz = metrics.reg_gompertz()'''
    #écriture du fichier    

    write_file(log_path, bdd, weight, metric, df_metrics, df_reglog, df_scope, layers, k, t0)    
#####################################################################################
