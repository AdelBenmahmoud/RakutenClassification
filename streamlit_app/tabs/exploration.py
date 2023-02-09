import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Exploration, visualisation et pre-processing des données"
sidebar_name = "Exploration des données"

super_categories = ['Livres', 'Jeux', 'Jouets', 'Meubles', 'Equipements', 'Deco', 'Autres']

df_X_train = pd.read_csv('../data/X_train_update.csv')
df_X_train = df_X_train.rename(columns={ df_X_train.columns[0]: "ID" })
df_Y_train = pd.read_csv('../data/Y_train_CVw08PX.csv')
df_Y_train = df_Y_train.rename(columns={ df_Y_train.columns[0]: "ID" })

def run():

    st.title(title)

    st.header('Présentation des données')
    st.markdown(
        """
        99 000 listes de produits au format CSV, y compris le train (84 916) et le test (13 812):
 
        - X_train.csv : fichier d'entrée d'entraînement
        - Y_train.csv : fichier de sortie d'entraînement
        - X_test.csv : fichier d'entrée de test

    
        """
    )

    st.caption('Premières lignes du fichier X_train.csv')
    st.dataframe(df_X_train.head())  
    
    st.caption('Premières lignes du fichier Y_train.csv')
    st.dataframe(df_Y_train.head())
        
    st.markdown(
        """
        « images.zip » contenant toutes les images (sous-dossiers image_train et image_test) avec les noms :
        - image_ **imageid** _ product _ **productid** .jpg


        """
    )


    st.header('Exploration des données ')
    st.markdown(
       """
       ### Variables explicatives

       Texte (**designation** + **description**):
       - Présence de NaN dans description 
       - Existence des descriptions très longues
   
       """
    )
    st.image(Image.open("assets/figure_variable_explicative_texte.png")) 

    st.markdown(
        """
        Quelques exemples des images :

        """
    )
    st.image(Image.open("assets/figure_variable_explicative_image.png")) 


    st.markdown(
       """
       ### Variables cibles:
    
       Code de produit (**prdtypecode**): 27 classes 
   
       """
    )
    st.image(Image.open("assets/figure_variable_cible_label.png")) 
    
    
    st.markdown(
       """
       ### Pre-processing des données textuelles 

    
        1. Nettoyage :
        - Enlever les doublons
        - Retirer les balises html
        - Transformer les phrases en liste de mots 
        - Convertir toutes les lettres en minuscules 
        - Supprimer les éléments de ponctuations 
        - Supprimer les balises, chiffres et autres caractères spéciaux 
        - Supprimer les stopwords
        
        2. Traduction des textes en français avec **googletrans**
        
        3. Ajout d’un champ supplementaire du texte avec stemmimg

   
       """
    )
    
        
    st.markdown(
       """
       ### Visualisation:
       - En super-catégorie (cf. tableau ci-dessous) 
       - Texte sans stemming
      
       """
    )
    
    supcat = st.selectbox("Choisissez la super-catégorie", super_categories)
        
    st.image(Image.open("assets/Capture_wordcloud_"+supcat+".png"))
    
    st.image(Image.open("assets/figure_tableau_super_categories.png")) 
    
    