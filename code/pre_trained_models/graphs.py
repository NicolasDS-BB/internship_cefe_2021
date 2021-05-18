#!/usr/bin/env python
#####################################################################################
# LIBRAIRIES:
#####################################################################################
import csv
import matplotlib.pyplot as plt
#####################################################################################
#SETTINGS:
#####################################################################################
model_name = 'VGG16'  # 'vgg16, resnet (...)'
#####################################################################################
#PATHS:
#####################################################################################
weights = 'imagenet'
log_path_cfd_imagenet ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/log_correlations_CFD'+'_'+weights+'L1.csv'
log_path_jen_imagenet ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/log_correlations_JEN'+'_'+weights+'L1.csv'
log_path_scut_imagenet ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/SCUT-FBP/log_correlations_SCUT-FBP'+'_'+weights+'L1.csv'
log_path_mart_imagenet ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/MART/log_correlations_MART'+'_'+weights+'L1.csv'
weights = 'vggface'
log_path_cfd_vggface ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/log_correlations_CFD'+'_'+weights+'L1.csv'
log_path_jen_vggface ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/JEN/log_correlations_JEN'+'_'+weights+'L1.csv'
log_path_scut_vggface ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/SCUT-FBP/log_correlations_SCUT-FBP'+'_'+weights+'L1.csv'
log_path_mart_vggface ='/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/MART/log_correlations_MART'+'_'+weights+'L1.csv'

log_path_cfd_vgg_face_tr_norm = '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/log_correlations_CFD_vggface_L1_TR.csv'
log_path_cfd_imagenet_tr_norm = '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/CFD/log_correlations_CFD_imagenet_L1_TR.csv'

log_path_mart_vgg_face_tr_norm = '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/MART/log_correlations_MART_vggface_L1_TR.csv'
log_path_mart_imagenet_tr_norm = '/home/nicolas/Bureau/internship_cefe_2021/data/redesigned/MART/log_correlations_MART_imagenet_L1_TR.csv'

#####################################################################################
# VARIABLES:
#####################################################################################
if model_name == 'VGG16':
        layers = ['in','b1_c1','b1_c2','b1_p','b2_c1', 'b2_c2','b2_p',
        'b3_c1','b3_c2','b3_c3','b3_p','b4_c1','b4_c2','b4_c3',
        'b4_p', 'b5_c1','b5_c2','b5_c3','b5_p','flat','fc1', 'fc2']   
elif model_name == 'resnet50':
    if weights == 'imagenet': 
        print('error, model not configured')
    elif weights == 'vggfaces':
        print('error, model not configured')
#####################################################################################
corr_cfd_imagenet = []
corr_scut_imagenet  = []
corr_mart_imagenet = []
corr_jen_imagenet = []
corr_cfd_vggface = []
corr_scut_vggface  = []
corr_mart_vggface = []
corr_jen_vggface = []

corr_cfd_vggface_tr = []
corr_cfd_imagenet_tr = []

corr_mart_vggface_tr = []
corr_mart_imagenet_tr = []
#####################################################################################
# PROCEDURE:
#####################################################################################
def parse_corr(corr, log_path, index_col = 1):
    '''
    parse csv file with correlations between layers sparseness and beauty/attractivity score
    '''
    with open (log_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter =';')
            index = 0
            for row in reader:
                index += 1            
                if index == 1:
                    continue
                if row == ['##############']:
                    break
                if not row:
                    continue           
                corr.append(float(row[index_col])*float(row[index_col])) #on mettrait au carré pour avoir les R2
#####################################################################################

#####################################################################################
# CODE:
##################################################################################### 
parse_corr(corr_cfd_vggface , log_path_cfd_vgg_face_tr_norm, index_col= 1)
parse_corr(corr_cfd_imagenet , log_path_cfd_imagenet_tr_norm, index_col= 1)
parse_corr(corr_cfd_vggface_tr , log_path_cfd_vgg_face_tr_norm, index_col= 3)
parse_corr(corr_cfd_imagenet_tr , log_path_cfd_imagenet_tr_norm, index_col= 3)
##################################################################################### 
parse_corr(corr_scut_imagenet, log_path_scut_imagenet)
parse_corr(corr_scut_vggface, log_path_scut_vggface )
##################################################################################### 
parse_corr(corr_mart_vggface, log_path_mart_vgg_face_tr_norm, index_col= 1)
parse_corr(corr_mart_imagenet, log_path_mart_imagenet_tr_norm, index_col= 1)
parse_corr(corr_mart_vggface_tr , log_path_mart_vgg_face_tr_norm, index_col= 3)
parse_corr(corr_mart_imagenet_tr , log_path_mart_imagenet_tr_norm, index_col= 3)
#####################################################################################    
Categories = ["CFD_imagenet","CFD_vggface","MART_imagenet","MART_vggface"] 

from math import * 
nb_categories = len(Categories)
largeur_barre = floor(1*10/nb_categories)/8
x1 = range(len(corr_cfd_imagenet))
x2 = [i + largeur_barre for i in x1]
x3 = [i + 2*largeur_barre for i in x1]
x4 = [i + 3*largeur_barre for i in x1]


plt.bar(x1, corr_cfd_imagenet_tr, width = largeur_barre, color = 'red',
           edgecolor = 'black', linewidth = 1)
plt.bar(x2, corr_cfd_vggface_tr, width = largeur_barre, color = 'darkred',
           edgecolor = 'black', linewidth = 1)
plt.bar(x3, corr_mart_imagenet_tr, width = largeur_barre, color = 'blue',
           edgecolor = 'black', linewidth = 1)
plt.bar(x4, corr_mart_vggface_tr, width = largeur_barre, color = 'darkblue',
           edgecolor = 'black', linewidth = 1)

plt.xticks([r + largeur_barre / nb_categories for r in range(len(corr_cfd_imagenet))],
              layers)

plt.legend(Categories,loc=2)
plt.show()



 #####################################################################################     
'''Categories = ["MART_imagenet_norm","MART_imagenet_TR"] 

from math import * 
nb_categories = len(Categories)
largeur_barre = floor(1*10/nb_categories)/10
x1 = range(len(corr_mart_imagenet))
x2 = [i + largeur_barre for i in x1]

plt.bar(x1, corr_mart_imagenet, width = largeur_barre, color = 'red',
           edgecolor = 'black', linewidth = 1)
plt.bar(x2, corr_mart_imagenet_tr, width = largeur_barre, color = 'blue',
           edgecolor = 'black', linewidth = 1)

plt.xticks([r + largeur_barre / nb_categories for r in range(len(corr_cfd_imagenet))],
              layers)

plt.legend(Categories,loc=2)
plt.show()'''
