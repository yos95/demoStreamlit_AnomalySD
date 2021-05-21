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
from glob import glob
from sklearn.metrics import f1_score
from tensorflow import keras
import numpy as np
import librosa



def logMelSpectrogram(audio, fe, dt):
    stfts = np.abs(librosa.stft(audio,n_fft = int(dt*fe),hop_length = int(dt*fe),center = True)).T
    num_spectrogram_bins = stfts.shape[-1]
    # MEL filter
    linear_to_mel_weight_matrix = librosa.filters.mel(sr=fe,n_fft=int(dt*fe) + 1,n_mels=num_spectrogram_bins,).T
    # Apply the filter to the spectrogram
    mel_spectrograms = np.tensordot(stfts,linear_to_mel_weight_matrix,1)
    return np.log(mel_spectrograms + 1e-6)




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
# page demo
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


    st.subheader("Détection d'anomalie par RNN")
    
    loaded_model = keras.models.load_model("Models/Classificateur_GRU256_"+e+"_5N.joblib")
    
    dt = 256
    freq = 256
    T_max=10
    machine=e
    path_model= loaded_model
    file_path= "dataset_lite/"+e+"/"+f
    file_name=f
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


  


 
# #############################################################################
# main
# #############################################################################
       
if __name__ == '__main__':
    main()
