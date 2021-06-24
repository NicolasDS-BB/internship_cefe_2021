#####################################################################################
# 1. DESCRIPTION:
#####################################################################################

#####################################################################################
# 2. LIBRAIRIES:
#####################################################################################
library("rjson")
library("purrr")

#####################################################################################
# 3. PARAMETRES: def analyse_metrics(model_name, bdd, weight, metric,k):
#####################################################################################

#mettre ça pas en dur a terme mais en paramètres passéa
setwd("/home/renoult/Bureau/internship_cefe_2021/code/functions")
model_name <- 'VGG16'
bdd <- 'CFD'
weight <- 'vggface'
metric <- 'L0'

#récupération du nom des couches en fonction du nom du modèle
if (model_name == 'VGG16'){
  
  if (weight == 'imagenet'){
    layers = c('input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
              'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
              'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2')} 
  else if (weight == 'vggface'){
    layers = c('input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3',
            'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','flatten',
            'fc6/relu','fc7/relu')}}

#path d'enregistrement des résultats et chargement des données  
if (bdd == 'CFD'){
  labels_path ='../../data/redesigned/CFD/labels_CFD.csv'
  log_path ='../../results/CFD/log_'

}else if (bdd == 'JEN'){
  labels_path ='../../data/redesigned/JEN/labels_JEN.csv'
  log_path ='../../results/JEN/log_'
  
}else if (bdd == 'SCUT-FBP'){
  labels_path ='../../data/redesigned/SCUT-FBP/labels_SCUT_FBP.csv'
  log_path ='../../results/SCUT-FBP/log_'

}else if (bdd == 'MART'){
  labels_path ='../../data/redesigned/MART/labels_MART.csv'
  log_path ='../../results/MART/log_'
  
}else if (bdd == 'SMALLTEST'){                       
  labels_path ='../../data/redesigned/small_test/labels_test.csv'
  log_path ='../../results/smalltest/log_'  

}else if (bdd == 'BIGTEST'){
  labels_path ='../../data/redesigned/big_test/labels_bigtest.csv'
  log_path ='../../results/bigtest/log_'  
}

matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))

#####################################################################################
# 4. ANALYSE:
#####################################################################################


#####################################################################################
# 4.1 Logistic regression:
#####################################################################################

df_metrics <- as.data.frame(matrix_metrics)

flatten_df_metrics = flatten(df_metrics[,-1])

vec = as.vector(flatten_df_metrics)

vmin = min(vec)



df_metrics = sapply(df_metrics, as.numeric)



vmin = min(as.list(flatten(df_metrics[,-1])))
vmax = max(as.list(flatten(df_metrics[,-1])))

if (metric %in% c('kurtosis','L0')){  
  
  vmin = min(as.list(flatten(df_metrics[,-1])))
  vmax = max(as.list(flatten(df_metrics[,-1])))
  
  df_metrics[,-1]= df_metrics[,-1] - vmin
  
}
  
  #transformation des valeurs pour qu'elles soient strictement positives    
  df_metrics = df_metrics.applymap(lambda x: x + abs(vmin)) 
  
  #transformation des valeurs pour qu'elles soient entre 0 et 1    
  df_metrics = df_metrics.applymap(lambda x: x/vmax) 
  
  
  

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  }







df_reglog = metrics.reglog(layers, df_metrics, dict_labels) 












#####################################################################################
# 4.2 Gompertz regression:
#####################################################################################
df_gompertz = metrics.reg_gompertz()

#####################################################################################
# 4.3 Difference between maximum and minimum
#####################################################################################
df_scope = metrics.minmax(df_metrics,dict_labels)    


#####################################################################################
# 4.4 Difference between maximum and minimum
#####################################################################################

#####################################################################################
# 4.5 Correlation between each metrics by layer and attractivness
#####################################################################################


#####################################################################################
# 5. ENREGISTREMENT DES RESULTATS:
#####################################################################################

#écrire le log
#write_file(log_path, bdd, weight, metric, df_metrics, df_reglog, df_scope, layers, k) 












