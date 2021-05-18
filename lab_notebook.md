# Lab notebook

## 22/03/2021

- réunion avec Julien: mise en place du stage, cf CR de réunion dans /meetings

## 23/03/2021

- convention signée par le CNRS: ne manque plus que la fac
- création de l'environnement de travail, un github

### BDD
- Jen: demande envoyée, en attente d'une réponse
- Chicago: demande envoyée, en attente d'une réponse
- MART: introuvable
- SCUT-FBP: téléchargée (227 MO)

### Sonia
- Architecture VGGface: https://github.com/rcmalli/keras-vggface
- son git: https://github.com/soniamaitieo/vggface_tripletloss
- plein d'autres trucs: cf mails du 23/03

### Alexandre
- coefficient de Gini pour mesuré la sparsité
- Hurley and Richard, 2008, IEEE --> introuvable

## 24/03/2021

### BDD
- Jen: 
	+ reçue, téléchargée
	+ liens de téléchargement des images, pas les images directement, à reprendre
	+ quelle différence entre les traits "beauty" et  "asesthetic quality" ? regarder biblio
	
- CFD: 
	+ reçue, téléchargée

- SCUT: 
	+ reçue, téléchargée

##25/03/2021

### BDD
- MART: fournie en local par Julien, introuvable sur internet

### Data management:

- création d'un répertoire dédié dans /code
		
- **CFD**: 
	+ *IMAGES*:supprimer les non neutres (N), regrouper toutes les images  dans le répertoire /data/redisigned/CFD/images (script: */code/data_management/data_management_pictures_CFD.py*) **DONE**
	+ *LABELS*  récuperer 2 colonnes:  ID,  Attractive, seulement pour les personnes se trouvant dans les images séléctionnées au point précédent (script: */code/data_management/				data_management_labels_CFD.py*) **DONE**
	
###26/03/2021

**SCUT-FBP**: nouvelle base de données trouvée, mise a jour avec 5500 images (majorité d'asiatiques, environ 10% de caucasiens dont Emma Watson)
	
###Data management:
	
- **SCUT-FBP**: 
	+ *IMAGES*:copier/coller dans le  dans le répertoire /data/redisigned/SCUT_FBP/images (script: */code/data_management/data_management_pictures_SCUT_FPB.py*) **DONE**
	+ *LABELS*  faire la moyenne des notes de tous les évaluateurs pour chaque image, créer un nouveau csv disponible dans*/data/redesigned/SCUT_FBP/labels_SCUT_FBP.csv. (script: */code/data_management/SCUT_FBP/data_management_labels_SCUT_FBP.py) **DONE**
	tw
	
##29/03/2021	

	- lecture d'articles, pc trop lent
##30/03/2021

Journée scientifique IA et biodiversité: https://www.labex-cemeb.org/fr/actualites/journee-scientifique-ia-et-biodiversite-30-mars-2021
--> similarity search comme plantnet? 

		
###Data management:
	
- **MART** : quel critère d'évaluation?  sur combien est-ce noté? 
--> *"The judgements were given according to a Likert scale of 1-to-7-points, where number 1 meant a highly negative emotion and 7 meant a highly positive emotion"*
à standardiser sur 5


	
	(...)
			
- **lire ça** :  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3330757/ *(irm sur perception esthétique)*
- **et ça** : https://www.nature.com/articles/s41593-021-00822-8
- **et ça**  :  https://www.sciencedirect.com/science/article/pii/S1077314220300333





# 01/04/2021

## réunion avec julien:

- étude de connectivité fonctionelle cérébrale (functionnal connectivity) qui mesure a quel point l'activation d'une région est éterminée par l'activité d'une autre région (= fkux d'infirmation dans e cerveau)  --> denrière veille: il y a 6 mois donc a voir

-  MEG: bonne réslution temporelle mais  mauvaise spatiale; IRM: inverse

- publi intéressants en neuroscieces sur l'efficacité de codage


## imageNet

récupération d'un réseau avec les poids d'imagenet: VGGface, suivant ceci: https://github.com/rcmalli/keras-vggface


vggface tuto: https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/



#  02/04/2021

- étude des exemples de keract (/keract/examples/vgg16.py ET heat_map.py

--> il faut maintenant l'appliquer a notre BDD et tester comment récupérer les valeurs d'activation
--> chercher aussi a quoi correspondent précisément les *gradients* de keract


# 06/04/2021

- création d'un groupe zotero pour la biblio
- lecture doc keract
- problème environnement python vscode


#  07/04/2021

- finalisation démarches carte accès campus
- résolution problème  vs code
- prise en main de keract: 
	+ heatmap des données d'activation sur une image test --> très lent mais fonctionnel
	+ autre fonction de heatmap liée a skl, ne marche pas: pourquoi? qu'est elle sensée faire exactement? 


# 08/04/2021

- récupération carte accès campus
-lecture de l'introduction (1.) de la publication sur la comlexité visuelle
- publi complexité visuelle: technique utilisée pour réduire les biais des notes: *"In order to minimize the potential bias from individual ground-truth contributors and limited rating scales, we obtained the ground-truth labels using a forced-choice pairwise crowdsourcing methodology. Labels were obtained from 1,687 contributors on more than 37,000 pairs of images. The pairwise scores were then converted to absolute scores on a [0,100] rating scale using the Bradley–Terry (Bradley and Terry, 1952) and matrix completion (Candès and Recht, 2009) methods."*
- calcul des activations: array a 4 dimensions, discuter demain lors de la réunion de quelle approche avoir pour le calcul de la norme l1 et d'autres métriques (quelle structure de données? )
- passage possible (testé) en keras sans keract: performances et résultats équivalents, code a peine plus long. 

# 09/04/2021

- **9h**: réunion avec sonia et julien, bilan dans /meetings_reports/meeting_09_04.md

- codage de la norme L2 d'une couche quelconque sur une image test

# 12/04/2021

- nettoyage "a la main" des labels CFD
- renommage adapté aux label des images CFD
- séléction des couches d'intérêt pour le calcul de norme (cf consignes julien)

# 13/04/2021

- détéction des facteurs de lenteur du programme: import de keract et chargement du modèle --> rien qui n'impact directement l'extraction des activations donc pas de soucis (en théorie) dans une boucle --> on verra lors du test

- comment gérer l'allumage du CPU les jours ou je suis en télétravail? : le laisser tout le temps allumé

- bouclage sur toutes les images (a faire avec le CPU):
	+ adaptation de la base de données: les labels sont ils exactement identiques auxnoms des fichiers? (fait pour CFD, fait pour MART, déja le cas pour SCUT-FBP)
	+ créer un dossier avec un csv avec toutes les notes, indépendament de l'origine (pas de dossier avec toutes les images, elles seront récupérées dans les 3 rpertoires des bdd pour éviter les doublons)

- mail envoyé au Dr Saraee pour avoir les scripts de sa publi sur la compexité visuelle
- Réunion avec sonia et julien: discussion sur la méthode utilisée dans la publi sur la complexité
- comment agréger toutes les normes d'une même image? : ne pas les agréger, raisonner couche par couche
- ticket envoyé a yoann pour avoir les id root du gpu situés dans le cahier de l'ancien stagiaire qui l'a installé 

# 14/04/2021

- passage en fonctionnel (en procédural en l'occurence) du code l1_norm_activation de manière a réuttiliser les mêmes fonctions sur les différents jeux de données pour les aggréger

- choix d'un dataframe pandas en tant que structure de données dédiée au calcul des corrélation norme l1 des couches/score de beauté car il s'avère
que c'est plus optimal qu'un dict dans ce cas précis, en terme de rapidité (je ne m'y attendait pas mais cette discussion argumente en ce sens: https://www.javaer101.com/fr/article/3415827.html)

# 15/04/2021

- finalisation du code des corrélations L1/score d'attractivité
- calcul des corrélations sur toute les bases de données mergées: elles tournent autour de 0 donc rien d'intéressant

# 16/04/2021

- réunion avec Julien et Sonia pour discuter des résultats: cf compte rendu de réunion dispo dans le dossier éponyme(qu'il faut que je remette au propre d'ailleurs)

- calcul des corrélations sur chaque base de données individuellement: rien d'intéressant sur scut_fbp et mart, R = 0,3 sur CFD

# 19/04/2021

- finalisation de la connexion a distance: réinstallation de Debian sur le pc gpu afin de récupérer les id root (dispos sur un post it sur le pc, dans les notes de mon téléphone ou dans un mail envoyé a Julien et Sonia)

- le mandrill était en fait séquencé

# 20/04/2021

- résolution d'un problème avec la connexion a distance: il faut créer des paramètres pré enregistrés avec un nom (ici CEFE) sur putty, au lieu de re rentrer les paramère a chaque fois, auquel cas ça ne fonctionnne pas. 

- recherche de thèse:
	+ téléchargement du dossier récap des ANR 2020-2021
	+ epfl: prochain sujets de thèse disponibles fin septembre. 
	+ sfbi: toujours que dalle

-envoie d'un mail aux auteurs de CFD pour aveoir le sexe du jury

# 21/04/2021

- retour négatif des auteurs de CFD (ils n'ont pas collecté cette donnée)
- journée peu productive

# 22/04/2021

- finalisation de la publi sur les approches Sheringtoniennes et Hopfieldiennes: concepts très intéressants mais bcp de blabla. 
- début de la lecture de la publi sur les putatives ratio
- relance de l'autrice de la publi sur la complexité des images pour obtenir son code. : nouveau mail aux co autrices jalal@cs.wisc.edu betke@bu.edu

# 23/04/2021

- lectures 2 publis:
	+ kriegeskorte
	+ ? 
	+ Putative ratio: pas convaincant, ne considère pas le images en général mais seulement les rations l/L de quelques rectangles

- création de sous dossier sur le repo zotero

- écoute de ce podcast: https://avisdexperts.ch/videos/view/9379

# 26/04/2021
 semil
- réponses de la co-autrice (Mona Jalal) de la publi sur la complexité --> la première autrice serait en train de réécrite le code et me le renverra via github (a suivre) 
- début de la lecture de la publi sur les matrices de dissimilarité: toolbox matlab dispo, recodage python plus simple? 

# 27/04/2019

- lecture publi matrice dissimilarité --> en fait peu d'intérêt
- lecture tuto matlab: http://www.lps.ens.fr/~krzakala/matlab.pdf
- réu avec sonia et julien 
- code d'une fonction pour calculer la moyenne des normes L1 des valeurs d'activation deschannels d'une couche
- adaptation globale du code pour passer en paramètre si l'on veut un calcul en flatten ou en moyenne de channel pour chaque couche


# 28/04/2021

- journée peu productive

# 29/04/2021

- importer les images de jen: dev d'un code fonctionnel mais le réseau du CNRS bloque: le faire tourner a la maison
- création de paramètres au début du programme facilitant son utilisation (poids, model, bdd etc... )

# 30/04/2021
 
- optimisation du code, petites modifications par ci par là (amélioration du log, etc...)

# 03/04/2021

- téléchargement de JEN en ayant géré les échecs (65 en l'occurence, c'est acceptable sur 1600 images)
- amélioration des paramètres du code des activations
- erreur avec VGGFace a comprendre et corriger

# 04 au 10/04/2021

- vgg face implémenté: même architecture mais implémentation difféente (quelle idée) ce qui expliquait les soucis
- il y avait une erreur de dimension concernant les channels: erreur corrigée
- suite a cette correction, les activations avec les méthodes channel et flatte sont en fait les mêmes: les channels ne sont donc pas pondérés pour vgg
--> n'uttiliser que flatten du coup (pour vgg)
- implémentation de Treve-Rolls: fonctionnel mais hyper lent ce qui la rend inutilisable sur les vraies bdd hors test:
	+ essayer avec reduce(), conseillé par alexandre
	+ regarder sur 30 images si TR corrèle avec L1, sinon ça vaut pas le coup selon julien (moi je trouve que si
	)
- impossible d'utiliser un disque dur externe (pour importer les donées) sur le pc gpu --> peut être que NS sait
- discussion avec une étudiante en psycho qui avait une bdd potentiellement intéressante: en fait non. On retiendra qu'il y a un master IA-psycho à Chambéry. 
- résultats sur la norme L1 avec CFD (pas avec TR)
-JEN: images trop grosses, preprocess impossible, régler ça

# 12/04/2021

- graphe des corrélations entre score et norme L1 pour toutes les bases de données et toutes les couches:
	+ avec le R (corrélation)
	+ avec aussi le R^2
--> résultats particulièrement intéressants avec CFD, moins avec les autres

# 13/04/2021

- développement d'une fonction treve-rolls alternative avec reduce(), mais les tests semblent incohérents, vérifier qu'elle fait bien ce qui est demandé
- création d'une bdd "bigtest" contenant 50 images de  
- calcul de la corrélation entre TR et L1 sur bigtest

# 14/04/2021

- idée pour faire fonctionner treve rolls avec reduce: tricker la récursivité en ne l'appliquant pas au terme A, et en le faisant en dehors de reduce pour l'index 0 du vecteur
- calcul de TR sur CFD avec imagenet et vgg face: résultats surprenants

# 17/04/2021

- découverte d'un dysfonctionnement de git
- calcul de TR sur MART avec imagenet et vggface: résultats encore plus surprenants
- graphes:
	+ **L1** Graphe sur toutes les bdd et tous les poids (un graphe rvalue, un graphe r2)
	+ **TR** Graphe sur toutes les bdd (que mart et cfd) et tous les poids (un graphe rvalue, un graphe r2)
	+ **comparaison** 2 graphe par bdd (r et r2) pour comparer L1 et TR


# 18/04/2021

- 



TODO:
- régler le souci avec git
- vérifier que la formule de TR choisie (la 2e de l'article) est vraiment mieux que la première
- faut il corriger les pvalues? Correction de Bonferroni adaptée? (car tests non indépendants)
- recoder Treve-Rolls avec reduce
- faire un fichier de labels propres pour JEN
- recoder preprocessed() avec tfdata (plus opti) et adapté au formats des images de JEN
- appliquer les conseils de Rufin, cf livre de géron, cf réunion du 08/04
- reproduire les résultats de la publi sur la complexité des imagees --> ou attendre que la coautrice les envoie comme elle s'y est engagée--> o
- Demander a alexandre des précisions sur la publi de Richard 2008 (erreur de recopiage de ma part?)
- écouter:
	+ https://avisdexperts.ch/videos/view/10980 

































