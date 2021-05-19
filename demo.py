#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# #############################################################################
# Projet DS - Sound Anomaly Detection - Promotion Bootcamp Mai 2020 - 
# DataScientest.com
# #############################################################################

import pandas as pd
import streamlit as st
import appsession as session
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
import numpy as np
import joblib
import librosa


def logMelSpectrogram(audio, fe, dt):
    stfts = np.abs(librosa.stft(audio,n_fft = int(dt*fe),hop_length = int(dt*fe),center = True)).T
    num_spectrogram_bins = stfts.shape[-1]
    # MEL filter
    linear_to_mel_weight_matrix = librosa.filters.mel(sr=fe,n_fft=int(dt*fe) + 1,n_mels=num_spectrogram_bins,).T
    # Apply the filter to the spectrogram
    mel_spectrograms = np.tensordot(stfts,linear_to_mel_weight_matrix,1)
    return np.log(mel_spectrograms + 1e-6)

# #############################################################################
# Parameters
# #############################################################################
#DATASET_FOLDER = "./dataset_lite" # folder with samples from each six engines: 
                                # 1. ToyCar (ID)
                                # 2. ToyConveyor (ID)
                                # 3. fan (ID)
                                # 4. pump (ID)
                                # 5. slider (ID)
                                # 6. valve (ID)
#DF = pd.read_feather(DATASET_FOLDER + "/df.feather").drop('label_e', axis=1)
#DF_RES = pd.read_feather(DATASET_FOLDER + "/df_test.feather")
SCORES = [[82.13, 85.76, 64.57, 86.33, 77.34, 65.28, 77.31, 55.31, 81.49, 
           63.37, 91.80, 67.20, 58.21, 97.64, 77.39, 98.93, 82.64, 94.40,
           75.22, 81.32, 76.13, 84.03, 69.58], # ROC-AUC scores par A-E
          [57.79, 81.85, 84.92, 67.72, 77.82, 58.00, 61.17, 53.57, 99.44,
           81.96, 91.64, 70.18, 78.40, 94.70, 72.17, 92.24, 79.83, 98.83,
           95.25, 90.42, 75.68, 74.01, 84.80]] # ROC-AUC scores par k-Means
# #############################################################################


def main():
    state = session._get_state()
    pages = {"Sound Anomaly Detection": page_dashboard,
             "Demo": page_demo}
    st.sidebar.title("Sound Anomaly Detection")
    st.sidebar.subheader("Menu")
    page = st.sidebar.radio("", tuple(pages.keys()))
    pages[page](state)
    state.sync()
    st.sidebar.info("Projet DS - Promotion Bootcamp Mai 2020"
                    "\n\n"
                    "Participants :"
                    "\n\n"
                    "[Corentin]\
                    (https://www.linkedin.com/in/corentindesmettre-9741951a/)"
                    "\n\n"
                    "[Astrid]\
                    (https://www.linkedin.com/in/astrid-lancrey/)"
                    "\n\n"
                    "[Youcef]\
                    (https://www.linkedin.com/in/youcef-madjidi/)"
                    "\n\n"
                    "[Yossef]\
                    (https://www.linkedin.com/in/yossef-aboukrat-39b90964/)"
                    )
    
    
# #############################################################################
# page HOME
# #############################################################################
        
def page_dashboard(state):
    st.title("Anomalous Sound Detection")
    st.header("Détection d'anomalies parmi des bruits de machines")
    st.write("\n\n")  
    st.write(
    " Le challenge DCASE2020 met au défi de trouver un modèle capable de prédire les anomalies générées par "
    " des machines d'usine sachant que le jeu de données est composé essentiellement de données audio normales. "
    " Le dataset dispose de quelques fichiers anormaux dans un dataset test pour chacune des 6 machines."
    " Les fichiers de chaque machine sont répartis sur trois ou quatre ID selon la machine. "
    "\n\n"
    "Le repo github complet est disponible [ici]\
    (https://github.com/yos95/demoStreamlit_AnomalySD).", 
    unsafe_allow_html=True)  


# #############################################################################
# page Dataset
# #############################################################################
def page_dataset(state):
    st.title("Le dataset")
    st.header("Données audio disponibles")
    st.write(
    "Le dataset que nous utilisons est celui qui a été fourni dans le cadre du\
    challenge DCASE 2020. Une copie non officielle de ces données est\
    accessible sur [kaggle](https://www.kaggle.com/daisukelab/dc2020task2)\
    sous licence Creative Common, et fait 10,12 Go."
    "\n\n"
    "Les données sont constituées de fichiers audio au format *.wav séparés en\
    6 dossiers, un par type de machine :"
    "\n\n"
    "• ToyCar"
    "\n\n"
    "• ToyConveyor"
    "\n\n"
    "• fan"
    "\n\n"
    "• pump"
    "\n\n"
    "• slider"
    "\n\n"
    "• valve"
    "\n\n"
    "Chaque type de machine est représenté par plusieurs machines,\
    différenciées par un identifiant indiqué dans le nom du fichier (ID dans\
    la suite)."
    "\n\n"
    "Chaque dossier est divisé en 2 sous-dossiers, Train et Test.", 
    unsafe_allow_html=True) 
    st.info(
    "Les dossiers Train ne contiennent que des données normales, alors\
    que les dossiers Test contiennent à la fois des données normales et des\
    données anormales.")        
    st.warning(
    "Chaque type de machine est représenté par 4 machines, à l'exception du\
    ToyConveyor, pour lequel on dispose d'enregistrements de 3 machines\
    seulement. \n\nCela fait un total de 23 machines différentes.")              


# #############################################################################
# page Exploration
# #############################################################################   
def page_analyse(state):    
    st.title("Analyse exploratoire des données")
    st.header("Le dataset DCASE 2020 task 2")    
    st.write("\n\n")
    st.write(
    "Les noms des fichiers audio sont rangés dans un DataFrame, et on leur\
    associe des informations qui peuvent s'avérer intéressantes :"
    "\n\n"
    "• le type de machine (engine)"
    "\n\n"
    "• l'ensemble auquel le fichier appartient (dir), train ou test"
    "\n\n"
    "• l'identifiant de la machine (ID)"
    "\n\n"
    "• la fréquence d'échantillonnage en Hz (fs)"
    "\n\n"
    "• la durée de l'enregistement en secondes (length)"
    "\n\n"
    "• le label de l'enregistrement (label) : 1 s'il s'agit d'un bruit normal,\
    0 sinon.")
    st.dataframe(DF.sample(frac=1).reset_index(drop=True).head(10))
    if st.checkbox("Afficher quelques informations complémentaires"):
        info_load_state = st.text('Loading data...')
        st.write(DF.groupby(['engine']).fs.unique())
        st.info("Tous les fichiers sont enregistrés à la même fréquence\
                d'échantillonnage (16 kHz).")
        st.dataframe(DF[['length',
                         'engine']].groupby(['engine'])['length'].unique())
        st.warning("On note que tous les enregistrements durent 10 s, à\
                   l'exception de ceux du ToyCar qui durent 11 s.")
        info_load_state.text("")
    if st.checkbox("Afficher les dimensions "):
        st.write(DF.shape)
        st.info("Notre dataset contient {} observations.".format(DF.shape[0]))
    img = Image.open("images/engines_repartition.png")
    st.image(img, width = 600, caption = "")
    st.info("Les données sont globalement équilibrées.")  
        

# #############################################################################
# page Transformation
# #############################################################################   
def page_transformation(state):    
    st.title("Transformation des données audio")
    st.subheader("Le problème")    
    st.write("\n\n")
    st.write(
    "La représentation temporelle des fichiers audio (voir page précédente)\
    reflète les variations de pression acoustique, mais ne transmet que peu\
    d'information en l'état. Les variations du contenu fréquentiel (graves,\
    aigus) au cours du temps sont bien exploitables. Quelques transformations\
    permettent d'obtenir ces informations.")
    st.write("\n\n")
    e = st.selectbox("Choisir une machine :", DF.engine.unique())
    d = st.checkbox("Bruit anormal")
    files = glob(DATASET_FOLDER+'/'+e+'/*.wav')
    f = [file for file in files if ('anomaly' in file)==d][0]
    st.write("<u>Fichier :</u>", f.split("/")[-1], unsafe_allow_html=True)    
    st.write("\n\n")  
    st.audio(f)
    st.write("\n\n")  
    st.write("<u>Signal audio :</u>", unsafe_allow_html=True)
    tplot_load_state = st.text('Loading plot...')
    fig_tp, s_tp = fct.plot_audio(f)
    st.pyplot(fig_tp)
    tplot_load_state.text("")
    st.write("Dimensions des données :", s_tp)
    st.subheader("Le spectrogramme fréquentiel")
    st.write("\n\n")
    st.write(
    "Cette représentation est obtenue en appliquant des FFT sur des fenêtres\
    temporelles de petites tailles (de l'ordre de 30ms), et reflète\
    directement les variations fréquentielles en fonction du temps.")
    splot_load_state = st.text('Loading plot...')
    fig_sp, s_sp = fct.plot_Spectrogram(f)
    st.pyplot(fig_sp)
    splot_load_state.text("")
    st.write("Dimensions des données :", s_sp)
    st.write("soit {} features.".format(s_sp[0]*s_sp[1]))
    st.info("On voit qu'avec un nombre de features quasiment équivalent, la\
            représentation est bien plus éloquente.")
    st.subheader("Les MFEC")
    st.write("\n\n")
    st.write(
    "Cette dernière représentation suit le même principe que la précédente,\
    mais on ajoute une étape de filtrage pour passer de l'échelle hertzienne\
    à l'échelle des Mel. Le rapport logarithmique entre les deux permet de\
    favoriser ce qui se passe dans les graves par rapport aux aigus (le seuil\
    se situant à 1kHz). Voyons quel est l'impact sur nos signaux :")
    mplot_load_state = st.text('Loading plot...')
    fig_mp, s_mp = fct.plot_MFEC_Spectrogram(f)
    st.pyplot(fig_mp)
    mplot_load_state.text("")
    st.write("Dimensions des données :", s_mp)
    st.write("soit {} features.".format(s_mp[0]*s_mp[1]))
    st.info(
    "On a divisé le nombre de features par 4. Cette représentation, bien que\
    différente de la précédente, ne semble pas porter atteinte à la quantité\
    d'informations.")

    
# #############################################################################
# page Méthodologie
# #############################################################################
def page_methodologie(state):
    st.title("Méthodologie")
    st.header("Approche")
    st.write(
    "Notre projet correspond à une problématique de détection d'anomalies. Il\
    s'agit donc de mettre en oeuvre un algorithme de clustering pour séparer\
    les anomalies des données normales."
    "\n\n"
    "Une autre méthode consiste à utiliser un auto-encodeur entrainé sur des\
    données normales, pour discriminer les anomalies en fonction de la\
    difficulté du modèle à les reconstruire correctement.")
    st.header("Préparation des fichiers audio")
    st.write(
    "\n\n"
    "Pour permettre une meilleure exploitation des fichiers audio par nos\
    différents algorithmes de Deep Learning, il est nécessaire de les préparer\
    ainsi :"
    "\n\n"
    "1. Homogénéisation des fichiers en les raccourcissant tous à 10 s"
    "\n\n"
    "2. Transformation en représentation log-MFEC (voir page précédente)")
    if st.checkbox("Infos"):
        st.info("On paramètre la transformation avec 1024 points pour la FFT\
                et un overlap de 50%.")
    st.write(
    "\n\n"
    "3. Normalisation des données (qui sont centrées réduites) par rapport à\
    l'ensemble Train"
    "\n\n"
    "4. Pour l'usage de l'auto-encodeur, les spectrogrammes sont découpés en\
    frames pour augmenter ses performances.")
    if st.checkbox("Infos "):
        st.info("On utilise des fenêtres de 32 échantillons (près d'une\
                seconde), qui se recouvrent sur 4 échantillons (environ 100\
                ms). Chaque signal de 10 s est ainsi transformé en 11 frames.")


# #############################################################################
# page Modélisation
# #############################################################################
def page_modelisation(state):
    st.title("Modélisation")
    st.info("Nous proposons deux méthodes pour détecter les anomalies dans\
            nos données audio.")
    # 1. CNN ##################################################################
    st.header("1. Classifieur CNN")
    st.write(
    "Nous utilisons un modèle CNN convolutif pour classifier l'ensemble des\
    bruits en 23 classes. C'est une mise en oeuvre quelque peu modifiée du\
    modèle détaillé [ici](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Primus_36_t2.pdf)." 
             "\n\n" , unsafe_allow_html=True)    
    st.subheader("Architecture")
    st.write("\n\n")          
    if st.checkbox("Afficher la description"):    
        st.write("<b>Partie 1 : extraction des features</b>"
        "\n\n"
        "Cette partie vise à extraire les caractéristiques importantes des\
        données audio. Elle réduit les dimensions des données de 128x313 à 64\
        valeurs, et est composée de :"
        "\n\n"
        "•	16 couches convolutives Conv1D d'au moins 128 filtres"
        "\n\n"             
        "•	suivies d'autant de couches de normalisation BatchNormalization"
        "\n\n"             
        "•	autant de couches d'activation LeakyReLU"
        "\n\n"   
        "•	autant de couches de Dropout à 20 %"
        "\n\n"             
        "•	3 couches de MaxPooling1D intercalées"
        "\n\n"             
        "•	une couche de GlobalAveragePooling1D pour finir, ramenant la\
        dimension de l'embedding à un vecteur de longueur 64."
        "\n\n"             
        "<b>Partie 2 : classification</b>"
        "\n\n"             
        "•	une couche dense à 64 filtres"
        "\n\n"    
        "•	un Dropout avec 20% d'éléments éliminés"
         "\n\n"    
        "•	une fonction d'activation LeakyReLU"
         "\n\n"    
        "•	une dernière couche dense avec autant de neurones que de classes,\
        à savoir 23.", unsafe_allow_html=True)    
    st.subheader("Entraînement")
    st.write("On sépare les données Train en un ensemble d'entrainement, et un\
             ensemble de validation (20 %). "
             "Pour entrainer ce modèle, on utilise 2 callbacks :"
             "\n\n"             
             "•	un EarlyStopping qui stoppe l'entrainement lorsque la fonction\
             de coût de validation ne s'améliore pas depuis 10 époques"
             "\n\n"             
             "•	un LearningRateScheduler pour réduire le taux d'apprentissage\
             au cours de l'entrainement : 0,001 pour les 50 premières époques,\
             puis réduit de 1,5 % par époque."
             "\n\n" , unsafe_allow_html=True)      
    st.subheader("Évaluation ")
    st.write("On valide notre modèle en observant comment il classe les\
             données test, à savoir un mélange de données normales et\
             anormales."
             "\n\n" , unsafe_allow_html=True)
    if st.checkbox("Afficher la matrice de confusion"): 
        img = Image.open("images/conf_matrix.png")
        st.image(img, width = 600, caption = "")       
    st.write("On mesure un f1-score de 78 %, ce qui n'est pas si mal dans la\
             mesure où notre jeu de données contient des anomalies."
             "\n\n" , unsafe_allow_html=True)   
    st.subheader("Détection d'anomalies ")
    st.write("On peut désormais utiliser l'embedding pour tenter de détecter\
             les anomalies de notre jeu de données."
             "\n\n"
             "A partir de ce vecteur de dimensions réduites, on met en oeuvre\
             un algorithme de clustering k-Means. On obtient ainsi un f1-score\
             moyen de 75 % pour la détection d'anomalie, qui se répartit\
             comme suit :"
             "\n\n" , unsafe_allow_html=True)
    img = Image.open("images/scores_kmeans.png")
    st.image(img, width = 250, caption = "")
    st.write("On observe de grands écarts dans ces résultats, qui vont de 50 %\
             à 98 % suivant la machine."
             "\n\n" , unsafe_allow_html=True)
    # 2. AUTO-ENCODER #########################################################
    st.header("2. Détection d'anomalie par auto-encodeur convolutif")  
    st.write(
    "Nous utilisons ici un modèle d'auto-encodeur convolutif (inspiré de\
    [celui-là](https://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Pilastri_86_t2.pdf)).\
    Le principe de l'auto-encodeur est de compresser les données d'entrée en\
    un vecteur latent de faible dimension (partie encodeur), puis de\
    reconstruire la donnée d'entrée à partir du vecteur latent (partie\
    décodeur). En entrainant un auto-encodeur à minimiser l'erreur de\
    reconstruction sur des données normales, il est établi qu'il aura plus de\
    difficulté à reconstruire des anomalies, permettant ainsi de les\
    distinguer.")
    st.subheader("Architecture")
    st.write("\n\n")
    img = Image.open("images/AE.png")
    st.image(img, width = 600, caption = "")
    if st.checkbox("Afficher la description "):    
        st.write("L’auto-encodeur se décompose en 2 parties :"
        "\n\n"
        "<b>Partie 1 :</b> la partie encodeur"
        "\n\n"      
        "Cette première partie a pour objectif de réduire les dimensions des\
        données d'entrée, par une succession de 5 couches Conv2D, chacune\
        étant suivie d'une BatchNormalization, d'une activation ReLU, et d'un\
        Dropout de 20 %."
        "\n\n"      
        "<b>Partie 2 :</b> la partie décodeur"
        "\n\n"
        "Cette seconde partie doit reconstruire les données d'entrée. Elle est\
        donc conçue comme transformation inverse de la partie encodeur, en\
        remplaçant les Conv2D par des Conv2DTranspose.",
        unsafe_allow_html=True)
    st.subheader("Entraînement")
    st.warning("Il est nécessaire d'entrainer un auto-encodeur par machine.")
    st.write("L'entrainement est effectué sur l'ensemble Train de chacune des\
             23 machines, et validé sur l'ensemble Test."
             "\n\n" 
             "Ici encore, on utilise 2 callbacks :"
             "\n\n"             
             "•	un EarlyStopping qui stoppe l'entrainement lorsque la fonction\
             de coût de validation ne s'améliore pas depuis 10 époques"
             "\n\n"             
             "•	un LearningRateScheduler pour réduire le taux d'apprentissage\
             au cours de l'entrainement : 0,001 pour les 30 premières époques,\
             puis réduit de 1 % par époque."
             "\n\n" , unsafe_allow_html=True)      
    st.subheader("Évaluation ")
    st.write("On valide notre modèle en comparant les scores ROC-AUC que l'on\
             obtient à ceux obtenus par le modèle précédent."
             "\n\n" , unsafe_allow_html=True)
    img = Image.open("images/scores_comparatifs.png")
    st.image(img, width = 350, caption = "")       
    st.write("On peut comparer les scores moyens de chaque méthode :"
             "\n\n"             
             "•	Moyenne CNN : 79.23"
             "\n\n"             
             "•	Moyenne AE : 77.97"
             "\n\n"             
             "•	Moyenne baseline : 73.55"
             "\n\n"             
             "On constate que les scores du premier modèle sont globalement\
             meilleurs, bien que l'auto-encodeur soit plus performant sur\
             la plupart des machines."
             "\n\n" , unsafe_allow_html=True)   
    st.info("La baseline est constituée des scores initiaux présentés par les\
            organisateurs du challenge.")


# #############################################################################
# page PREDICTION
# #############################################################################
def page_demo(state):
    df = pd.read_csv("Fichiers_son.csv")
    st.set_option('deprecation.showfileUploaderEncoding', False)                                       
    st.title("Prédiction (Démo)")
    st.subheader("Choix de la machine")
    engine = list(df.Machine.unique())
    e = st.selectbox("Choix de la machine", engine)
    st.subheader("Choix du fichier")
    files = glob("dataset_lite"+'/'+e+'/*.wav')
    file_names = [f.split('/')[-1] for f in files]
    f = st.selectbox("Choix du fichier", file_names)    
    
    st.audio(files[file_names.index(f)])

    ID = f.split('_')[2]
    #lu = DF_RES[(DF_RES.engine==e) & (DF_RES.ID==ID)].label_e.unique()[0]

    # 1. RNN ##################################################################
    st.subheader("Détection d'anomalie par RNN")
    
    #news_vectorizer = open("./Classificateur_slider_LTSM_0515_V2.pkl","rb")
    #news_cv = joblib.load(news_vectorizer)
    loaded_model = keras.models.load_model("Models/Classificateur_GRU256_"+e+"_5N.joblib")
    
    #CODE PREDICTION STREAMLIT
    dt = 256
    freq = 256
    T_max=10
    machine='ToyCar' 
    path_model= loaded_model
    file_path= "dataset_lite/"+e+"/"+f
    file_name="anomaly_id_01_00000011.wav"
    #constantes
    seuil_machine={'ToyCar':[0.5,0.5,0.5,0.95,0.1],'ToyConveyor':[0.5,0.5,0.95,0.07],'fan':[0.5,0.5,0.5,0.95,0.1],'pump':[0.5,0.5,0.5,0.95,0.07],'slider':[0.5,0.95,0.5,0.5,0.05],'valve':[0.5,0.5,0.5,0.95,0.17]}
    ID_machine={'ToyCar':["ID_2","ID_3","ID_4","ID_1"],'ToyConveyor':["ID_2","ID_3","ID_1"],'fan':["ID_4","ID_6","ID_0","ID_2"],'pump':["ID_0","ID_2","ID_4","ID_6"],'slider':["ID_0","ID_2","ID_4","ID_6"],'valve':["ID_0","ID_2","ID_4","ID_6"]}
    print('Récupération des données audio')
    data, fe = librosa.load(file_path, sr=None)
    # For the audio > T_max : Use just the fist T_max seconde, to have the right shape. 
    if len(data)>= T_max*fe:
        data = data[:int(T_max*fe)]
    # For the audio < T_max : Add in the signal a zeros vector, to have the right shape.
    else :
        data = np.concatenate([data, np.zeros(int(T_max*fe - len(data)))])
    # Apply the logMelSpectrogram function. 
    audio_to_pred = logMelSpectrogram(data[:(len(data)//dt)*dt], fe, np.around(T_max/dt,3))
    audio_to_pred=audio_to_pred[:dt,:] # on tronque quand même au cas où nous avons dépasser le nombre de dt. 
    audio_to_pred=(np.array(audio_to_pred)-np.array(audio_to_pred).min())/(np.array(audio_to_pred).max()-np.array(audio_to_pred).min())
    # on cut pour avoir le bon nombre de fréquences
    selector=np.where(audio_to_pred.max(axis=0)>np.sort(audio_to_pred.max(axis=0))[-(freq+1)],True,False) # on est très restrictif dans un premier temps 
    if np.sum(selector)<freq:
        selector=np.where(audio_to_pred.max(axis=0)>=np.sort(audio_to_pred.max(axis=0))[-(freq+1)],True,False) # si nous avons été trop restrictif, alors nous élargissons la selection puis nous bouclons pour supprimer le surplus
    iter=100
    while np.sum(selector)>freq and iter>0: # En cas de multiples valeurs similaires on boucle pour les supprimer et ne garder que la dim voulue
        print('Trop de fréquences choisies, suppression de {:} fréquences pour atteindre la taille cible'.format( selector.sum()-freq))
        selector[np.argsort(np.sort(audio_to_pred.max(axis=0)[selector]))[-1]]=False  # on récupère la position du plus petit élément de datamax pour la supprimer du sélector
        iter-=1 # on met 10 itérations pour éviter de boucler à l'infini en cas d'erreur
    audio_to_pred=audio_to_pred[:,selector].reshape(1,dt,freq)
    print('Chargement du modèle')
    Classificateur_GRU256=keras.models.load_model("Models/Classificateur_GRU256_"+e+"_5N.joblib")
    #Classificateur_GRU256=keras.models.load_model(path_model+"Classificateur_GRU256_"+e+"_5N.joblib")
    print('Prédictions')
    y_pred = Classificateur_GRU256.predict(audio_to_pred)
    y_pred_save = tuple(y_pred)
    y_pred=y_pred.reshape(y_pred.shape[1])
    # on réalise la prédiction d'ID
    pred_ID=ID_machine[machine][y_pred[:y_pred.shape[0]-1].argmax()]
    # on réalise la prédiction normal / anormal
    pred_anorm = ""
    if (seuil_machine[machine][y_pred[:y_pred.shape[0]-1].argmax()] > y_pred[y_pred[:y_pred.shape[0]-1].argmax()]) | (seuil_machine[machine][y_pred.shape[0]-1] < y_pred[y_pred.shape[0]-1]):
        pred_anorm = 'anomaly'
    else :
        pred_anorm = 'normal'

    true=df[(df.Machine== machine) & (df.fichier==file_name)]
    cross_ID= 'bonne ' if pred_ID=='ID_'+str(true.iloc[0,5]) else 'fausse'
    cross_norm= 'bonne ' if pred_anorm==true.iloc[0,6] else 'fausse'
    st.success('Données réelles : ' '\n' + machine + ' ID_'+str(true.iloc[0,5])+" "+true.iloc[0,6]+ '\n\n' 'Prédiction ' + "la machine "+ machine+ " "+ pred_ID + " à un son "+ pred_anorm + '\n'  "la prédiction d'ID est "+ cross_ID + ", la prédiction de normalité est " +cross_norm)
    #Model_RNN = keras.models.load_model("Models/Classificateur_GRU256_ToyCar_5N.joblib")
    #st.image(image, width=None)
    # 2. AE ##################################################################
   
    #Model_AE = keras.models.load_model("Models/Classificateur_GRU256_ToyCar_5N.joblib")
    #st.image(image, width=None)

  


# #############################################################################
# page CONCLUSION
# #############################################################################
def page_conclusion(state):
    st.title("Conclusion et perspectives")
    st.header("Bilan")
    st.write(
    "L'objectif de ce projet était de détecter des anomalies à partir des\
    bruits de différentes machines. Ce but a été atteint avec un score ROC-AUC\
    moyen de 79 % avec le classifieur CNN. Nous avons vu que l'auto-encodeur\
    était plus performant sur certaines machines : on pourrait donc améliorer\
    ce résultat en choisissant judicieusement le modèle en fonction de la\
    machine."
    "\n\n"
    "Pour obtenir ce résultat, les données ont été analysées, transformées, et\
    les modèles ajustés au mieux avec les moyens disponibles.",
    unsafe_allow_html=True)  
    st.header("Pistes d’améliorations")
    st.write(
    "Nous pensons que de meilleurs résultats pourraient être obtenus en\
    approfondissant la démarche d'optimisation des modèles, ce que nous\
    n'avons pu faire du fait de limitations techniques auxquelles nous étions\
    soumis.")
    st.header("Perspectives")     
    st.write(
    "Après amélioration des résultats, il serait pertinent de mettre en oeuvre\
    ce genre de solution de manière concrète pour la détection d'anomalie en\
    temps réel, afin d'optimiser les process industriels.",
    unsafe_allow_html=True)  
     
# #############################################################################
# main
# #############################################################################
       
if __name__ == '__main__':
    main()
