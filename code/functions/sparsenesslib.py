#!/usr/bin/env python

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
t0 = time.time()
from tensorflow.keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from scipy.stats import linregress
import sys
sys.path.insert(1,'../../code/functions')
import sparsenesslib as spl #personnal library
import statistics as st
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import seaborn as sns
import scipy.optimize as opt
#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def gini(vector):
    """Calculate the Gini coefficient of a numpy array."""    
    if np.amin(vector) < 0:
        # Values cannot be negative:
        vector -= np.amin(vector)
    # Values cannot be 0:     
    vector = [i+0.0000001 for i in vector]     
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
def compute_filter(activations, activations_dict,layer,formula,k):
    '''
    Compute chosen formula (gini, treve-rolls, l1 norm...) for each filter (flatten), and compute the mean of them
    '''    
    filter = []

    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)

    tuple = activations[layer].shape
    liste = list(tuple)    
    nb_channels = liste[3] 

    for each in range(0, nb_channels-1): #on itère sur chaque filtre (profondeur)
        filter.append([])
        index_row = 0
        for row in activations[layer][0]:
            index_column = 0
            for column in activations[layer][0][index_row]:            
                filter[each].append(activations[layer][0][index_row][index_column][each])                 
                index_column += 1
            index_row += 1

    filter_metrics = []

    for each in filter:
        if formula == 'L0':                
            filter_metrics.append(1 - (norm(each, 0)/len(each)))
        if formula == 'L1':                
            filter_metrics.append(norm(each, 1))
        elif formula == 'treve-rolls':
            filter_metrics.append(treves_rolls(each))
        elif formula == 'gini':            
            filter_metrics.append(gini(each))
        elif formula == 'kurtosis':
            filter_metrics.append(scipy.stats.kurtosis(each))
    activations_dict[layer] = st.mean(filter_metrics)

####################################################/length#################################
def compute_flatten(activations, activations_dict,layer,formula,k):
    '''
    Create a flatten vector from each layer and compute chosen formula (gini, treve-rolls, l1 norm...) on it"
    '''
    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)
    
    arr = activations[layer].flatten()
    if formula == 'L0':    
        activations_dict[layer] = (1 - (norm(arr, 0)/len(arr)))
    if formula == 'L1':    
        activations_dict[layer] = (norm(arr, 1))        
    elif formula == 'treve-rolls':        
        activations_dict[layer] = (treves_rolls(arr))
    elif formula == 'gini':
        activations_dict[layer] = (gini(arr))
    elif formula == 'kurtosis':
        activations_dict[layer] = (scipy.stats.kurtosis(arr))
#####################################################################################
def compute_channel(activations, activations_dict,layer,formula,k):
    '''
    Compute chosen formula (gini, treve-rolls, l1 norm...) for each channel, and compute the mean of them
    '''
    channels = []
    index_row = 0

    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)

    for row in activations[layer][0]:
        index_column = 0
        for column in activations[layer][0][index_row]:            
            channel = activations[layer][0][index_row][index_column]      
            if formula == 'L0':                
                channels.append(1-(norm(channel, 0)/len(channel)))                
            elif formula == 'L1':                
                channels.append(norm(channel, 1))
            elif formula == 'treve-rolls':
                channels.append(treves_rolls(channel))
            elif formula == 'gini':
                channels.append(gini(channel))
            elif formula == 'kurtosis':
                channels.append(scipy.stats.kurtosis(channel))
            index_column += 1
        index_row += 1    
    activations_dict[layer] = st.mean(channels)
#####################################################################################
def compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula, k):
        
        for layer in layers:            
            if computation == 'channel':                
                if layer in flatten_layers:
                    compute_flatten(activations, activations_dict, layer,formula,k)       
                else:                     
                    compute_channel(activations, activations_dict, layer, formula,k)
            elif computation == 'filter':
                if layer in flatten_layers:
                    compute_flatten(activations, activations_dict, layer,formula,k)       
                else:                     
                    compute_filter(activations, activations_dict, layer, formula,k)
            elif computation == 'flatten':
                compute_flatten(activations, activations_dict, layer, formula,k)            
                
            else: print('ERROR: computation setting isnt channel or flatten')
#####################################################################################
def compute_sparseness_metrics_activations(model, flatten_layers, path, dict_output, layers, computation, formula, freqmod,k):
    '''
    compute the l1 norm of the layers given in the list *layers*
    of the images contained in the directory *path*
    by one of those 2 modes: flatten or channel (cf previous functions)
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
            (1, image.shape[0], image.shape[1], image.shape[2]))  # ?
        image = preprocess_input(img)
        # récupération des activations
        activations = keract.get_activations(model, image)
        activations_dict = {}
        compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula,k)
        dict_output[each] = activations_dict
#####################################################################################
def parse_rates(labels_path , dict_labels):
    '''
    Stores notes and image names contained in *path* 
    in *dict* as {name:note}    
    '''
    with open(labels_path, newline='') as labels:
        reader = csv.reader(labels)
        for line in reader:
            key = line[0]
            rate = line[1]
            dict_labels[key] = float(rate)
#####################################################################################
def create_dataframe(dict_rates, dict_norm, name = 'rate'):
    '''
    Creates a pandas dataframe that has a beauty score associates
    the L1 of the associated image layers
    rows: images, column 1: notes, column 2 to n: L1 norms
    '''
    df1 = pandas.DataFrame.from_dict(dict_rates, orient='index', columns = [name])
    df2 = pandas.DataFrame.from_dict(dict_norm, orient='index')     

    df = pandas.concat([df1, df2], axis = 1)     
    return df
#####################################################################################
def create_dataframe_reglog(dict_rates, dict_norm, name = 'rate'):
    '''
    Creates a pandas dataframe that has a beauty score associates
    the L1 of the associated image layers
    rows: images, column 1: notes, column 2 to n: L1 norms
    '''
    df1 = pandas.DataFrame.from_dict(dict_rates, orient='index', columns = [name])
    df2 = pandas.DataFrame.from_dict(dict_norm, orient='index', columns = ['reglog'] )     

    df = pandas.concat([df1, df2], axis = 1)     
    return df
#####################################################################################
def gompertzFct (t , N , r , t0 ):
        return N * np . exp ( - np . exp ( - r * (t - t0 ))) 
#####################################################################################
def layers_analysis(bdd,weight,metric, model_name, computer, freqmod,k = 1):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*
    '''
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

    else: #all paths are relative paths
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
    #####################################################################################
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
    #####################################################################################
    # VARIABLES:
    #####################################################################################
    dict_compute_metric = {}
    dict_labels = {}
    #####################################################################################
    # CODE:
    #####################################################################################
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


    parse_rates(labels_path, dict_labels)
    df_metrics = spl.create_dataframe(dict_labels, dict_compute_metric)    
    #####################################################################################
    #écriture des histogrammes
    #####################################################################################   
    y = []    
    for layer in layers:
        if layer[0:5] == 'input':
            layer = 'input' + '_' + str(k)         
        y = list(itertools.chain(y, list(df_metrics[layer])))
    title = 'distrib_'+ bdd +'_'+ weight +'_'+ metric   
    plt.hist(y, bins = 40)        
    plt.title(title, fontsize=10)                 
    plt.savefig(log_path +'_'+ bdd +'_'+ weight +'_'+ metric +'.png')
    plt.clf()
    #####################################################################################
    #régression logistique
    ##################################################################################### 
    i = 1  
    x = []  
    for each in range(len(layers)):
        x.append(i)
        i += 1
    x = pandas.DataFrame(x) 

    dict_reglog = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        for each in list(row)[1]:
            if j != 0:
                y.append(each)
            j += 1   

        picture = list(row)[0]

        y= pandas.DataFrame(y)
        df = pandas.concat([x,y], axis=1)   
        df.columns = ['x', 'y']
        # on ajoute une colonne pour la constante
        x_stat = sm.add_constant(x)
        # on ajuste le modèle
        model = sm.Logit(y, x_stat)
        result = model.fit()    
        #on récupère le coefficient
        coeff = result.params[0]
        dict_reglog[picture] = coeff

    print(dict_reglog)       

    df_reglog = spl.create_dataframe_reglog(dict_labels, dict_reglog)
    '''sns.lmplot(x='x', logistic=True, y='y', data = df)
    plt.show()'''
    #####################################################################################
    #minimum - maximum
    ##################################################################################### 
    dict_scope = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        for each in list(row)[1]:
            if j != 0:
                y.append(each)
            j += 1   

        picture = list(row)[0]

        maximum = max(y)
        minimum = min(y)
        diff = maximum - minimum


        dict_scope[picture] = diff       

    df_scope = spl.create_dataframe_reglog(dict_labels, dict_scope)
    #####################################################################################
    #Gompertz
    #####################################################################################   
    '''I_t = y [ x :]
    t = np.arange (len( I_t ))

    model = gompertzFct ;
    guess = (100000. , .1 , 50.)

    parameters , variances = opt . curve_fit ( model , t , I_t , p0 = guess )

    G_t = model (t , * parameters )

    print ( np . sqrt ( np . mean (( I_t - G_t )**2)))'''
    #####################################################################################
    #écriture du fichier
    #####################################################################################
    with open(log_path +'_'+ bdd +'_'+ weight +'_'+ metric +'.csv',"w") as file:            
        file.write('layer')
        file.write(';')
        file.write('mean_'+str(metric)) #valeur moyenne de la métrique par couche
        file.write(';')
        file.write('sd_'+str(metric)) #écart type de la métrique par couche
        file.write(';')
        file.write('corr_beauty_VS_'+'metric') #corrélation de la métrique avec la valeur de beauté/attractivité
        file.write(';')
        file.write('pvalue')
        file.write(';')
        file.write('\n') 
        for layer in layers:
            if layer[0:5] == 'input':
                layer = 'input' + '_' + str(k)
            file.write(layer)
            file.write(';')
            #mean
            l1 = list(df_metrics[layer])
            file.write(str(st.mean(l1)))
            file.write(';')    
            #standard deviation
            l1 = list(df_metrics[layer])
            file.write(str(st.stdev(l1)))
            file.write(';') 
            #correlation with beauty
            l1 = list(df_metrics[layer])
            l2 = list(df_metrics['rate'])
            reg = linregress(l1,l2)
            coeff = str(reg.rvalue)         
            file.write(coeff)
            file.write(';')  
            #pvalue
            pvalue = str(reg.pvalue) 
            file.write(pvalue)
            file.write(';')  
            file.write("\n")        
        t1 = time.time()
        file.write("\n")
        file.write("##############")
        file.write("\n")
        file.write('bdd;'+ bdd)
        file.write("\n")
        file.write('weights;'+ weight)
        file.write("\n")        
        file.write('metric;'+ metric)
        file.write("\n")
        total = (t1-t0)/60
        total = str(total)
        file.write("time;")    
        file.write(total)
        file.write(';minutes')
        file.write('\n') 
        #correlation with scope
        l1 = list(df_scope['reglog'])
        l2 = list(df_scope['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        file.write('coeff_scope: ')     
        file.write(';')   
        file.write(coeff)        
        file.write('\n')  
        #correlation with coeff of logistic regression
        l1 = list(df_reglog['reglog'])
        l2 = list(df_reglog['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        file.write('coeff_scope: ')  
        file.write(';')     
        file.write(coeff)        
        file.write('\n') 
#####################################################################################
