import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Modélisation"
sidebar_name = "Modélisation"
models = ["Modèle unimodal texte (RNN embedding avec Word2Vec skip-gram)",
          "Modèle unimodal image (EfficientNetB1)",
          "Modèle multimodal (fusion au niveau de décision)"]

def run():

    st.title(title)
    
    
    st.header('Récapitulatif de performance des modèles texte')
    st.markdown(
        """
        ### Modèles classiques de Machine Learning
        
        - Vectorization des textes avec TfidfVectorizer 
        - Sur-apprentissage important
        - Reduction de nombre de vocabulaire et de temps de calcul avec stemming
        - Résultat non sensible à l’effet de stemming

        """
    )
    st.image(Image.open("assets/figure_recap_modele_text_ML.png"))
    
    st.markdown(
        """
        ### Modèles de Deep Learning
        
        - Vectorization des textes et transformation en sequence avec Tokenizer (tensorflow.keras.preprocessing.text) et pad_sequences (tensorflow.keras.preprocessing.sequence)
        - Encodage de variable cibles avec LabelEncoder
        - Taille d’embedding à 300
        - Embedding simple ou avec RNN: sur-apprentissage important
        - Embedding avec des modèles existants : Amélioration de score

        """
    )
    st.image(Image.open("assets/figure_recap_modele_text_DL.png"))


    st.header('Récapitulatif de performance des modèles image')  
    st.markdown(
        """
        - Generateur des images avec Flow_from_dataframe (keras.preprocessing.image.ImageDataGenerator)
        - Entrainement des modèles de Transfert Learning sur les échantillons des images
        - Sélection du meilleur modèle (EfficientNetB1)
        - Entrainement du meilleur modèle sur l’ensemble des images en variant batch_size et dimensions 

        """
    )
    st.image(Image.open("assets/figure_recap_modele_img.png"))
    st.image(Image.open("assets/figure_recap_modele_img_EfficientNet.png"))
        
        
    st.header('Fusion - Modèle multimodal')   
    st.markdown(
        """
        - Rechargement des meilleurs modèles de texte et d’image
        - Concaténation des propabilités des classes 
        - Prédiction finale avec réseau de neurones de 2 couches Dense 

        """
    )
    st.image(Image.open("assets/figure_modele_fusion.png"))
    
    st.header('Performance des modèles retenus')
       
    st.markdown("Le modèle agrégé multimodal a réussi à augmenter la performance de prédiction d'environ **7%** par rapport au modèle unimodal de texte.")

    st.image(Image.open("assets/figure_modeles_retenus_score.png"))

    model = st.selectbox("Choisissez le modèle", models)
    
    st.markdown(
        """
        Classification report :
        """
    )
    
    if model == models[0]:       
        st.image(Image.open("assets/Capture_classification_report_text_W2V.png"), width=300)
        st.image(Image.open("assets/confusion_matrix_recall_text_W2V.png"))
    if model == models[1]:
        st.image(Image.open("assets/Capture_classification_report_image_EfficientNet.png"), width=300)
        st.image(Image.open("assets/confusion_matrix_recall_image_EfficientNet.png"))
    if model == models[2]:
        st.image(Image.open("assets/Capture_classification_report_fusion.png"), width=300)
        st.image(Image.open("assets/confusion_matrix_recall_fusion.png"))


