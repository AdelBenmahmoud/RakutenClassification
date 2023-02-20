import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import string
import nltk
nltk.download('all')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


title = "Demo"
sidebar_name = "Demo"

models = ["Modèle unimodal texte (RNN embedding avec Word2Vec skip-gram)",
          "Modèle unimodal image (EfficientNetB1)",
          "Modèle multimodal (fusion au niveau de décision)"]

classes_dict = {10:' Livres occasion ',
                40:' Jeux vidéo, CDs, équipement, câbles, neufs ',
                50:' Accessoires gaming ',
                60:' Consoles de jeux ',
                1140:' Figurines, objets pop culture ',
                1160:' Cartes de jeux ',
                1180:' Figurines et jeux de rôles ',
                1280:' Jouets enfants ',
                1281:' Jeux société enfants ',
                1300:' Jeux techniques, équipement, petites machines ',
                1301:' Chaussettes bébés, petites photos ',
                1302:' Equipements, Habits, outils, jouets, objets sur fond blanc ',
                1320:' Puériculture, accessoire bébé ',
                1560:' Mobilier général : meuble, matelas, canapés, lampes, chaise ',
                1920:' Linge de maison, oreillers, coussins ',
                1940:' Confiseries ',
                2060:' Décoration ',
                2220:' Animalerie ',
                2280:' Journaux et revues occasion ',
                2403:' Livres, BD et magazines ',
                2462:' Jeux vidéo occasion ',
                2522:' Fournitures papeterie et accessoires bureau ',
                2582:' Mobilier de jardin : meubles et outils pour le jardin ',
                2583:' Piscines et accessoires ',
                2585:' Outillages de jardin, équipement technique extérieur maison et piscines ',
                2705:' Livres neufs ',
                2905:' Jeux vidéo pour PC '}

classes_list = ['Manuel', 
                '10   - Livres occasion ',
                '40   - Jeux vidéo, CDs, équipement, câbles, neufs ',
                '50   - Accessoires gaming ',
                '60   - Consoles de jeux ',
                '1140 - Figurines, objets pop culture ',
                '1160 - Cartes de jeux ',
                '1180 - Figurines et jeux de rôles ',
                '1280 - Jouets enfants ',
                '1281 - Jeux société enfants ',
                '1300 - Jeux techniques, équipement, petites machines ',
                '1301 - Chaussettes bébés, petites photos ',
                '1302 - Equipements, Habits, outils, jouets, objets sur fond blanc ',
                '1320 - Puériculture, accessoire bébé ',
                '1560 - Mobilier général : meuble, matelas, canapés, lampes, chaise ',
                '1920 - Linge de maison, oreillers, coussins ',
                '1940 - Confiseries ',
                '2060 - Décoration ',
                '2220 - Animalerie ',
                '2280 - Journaux et revues occasion ',
                '2403 - Livres, BD et magazines ',
                '2462 - Jeux vidéo occasion ',
                '2522 - Fournitures papeterie et accessoires bureau ',
                '2582 - Mobilier de jardin : meubles et outils pour le jardin ',
                '2583 - Piscines et accessoires ',
                '2585 - Outillages de jardin, équipement technique extérieur maison et piscines ',
                '2705 - Livres neufs ',
                '2905 - Jeux vidéo pour PC ']


stop_words = set(stopwords.words('french') + stopwords.words('english'))
mots_vides = ["x", "cm", "mm", "h", "g", "peut", "être", 'e',"l'",'x','p','re', 'li','x','b','d','h', 'pla','br','id','al','ra','pla','sine','r','g','v','u','f']
stop_words.update(mots_vides)
def word_split(text):
    """split text into words, remove non alphabetic tokens and stopwords"""   
    # suppression de la ponctuation
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(table)    
    # séparation des textes en listes de mots
    tokens = word_tokenize(text)   
    # conversion en minuscule
    tokens = [w.lower() for w in tokens]   
    # restriction aux charactères alphabétiques
    words = [word for word in tokens if word.isalpha()]
    # filtrage des stopwords
    words = [w for w in words if not w in stop_words]
    
    return words



#### import donnees preprocessing
df = pd.read_csv("streamlit_app/assets/rakuten_data_preproc.csv", index_col= 0)
traduction = pd.read_table("streamlit_app/assets/traduction_designation-description.txt", header=None, squeeze=True)
traduction.index = df.index
traduction = traduction.apply(word_split)
df['text_data'] = traduction

df_X_train = pd.read_csv('data/X_train_update.csv', index_col=0)

df = pd.merge(df, df_X_train, left_index=True, right_index=True)

#### preparation des donnees pour les modeles
X = df[['text_data', 'image', 'prdtypecode']]
y = df['prdtypecode']

encoder =  LabelEncoder()
Y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# tokenization pour les textes
X_train_txt = X_train['text_data'].apply(lambda x : " ".join(w for w in x))
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train_txt)
max_length = 450

# generateur des données d'image
X_train_img = X_train[['image', 'prdtypecode']]
X_train_img['prdtypecode2']=X_train_img['prdtypecode'].astype(str)

image_path="data/images/image_train"
width = 240
height = 240 # resize 
batch = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = None)                                                           
train_set = train_datagen.flow_from_dataframe(dataframe=X_train_img,
                                              directory=image_path,
                                              x_col = "image",
                                              y_col = 'prdtypecode2',
                                              seed=42,
                                              class_mode="sparse",
                                              target_size = (width, height),
                                              batch_size = batch,
                                              shuffle=False)

###############################################################################

def run():

    st.title(title)
    
    st.markdown(
        """
        Ceci est une demonstration pour tester la prediction des modèles unimodals et multimodal:
            
        """
    )
    
    model_txt = st.checkbox(models[0])
    model_img = st.checkbox(models[1])
    model_fus = st.checkbox(models[2])
    
     
    st.header('Prediction')
    
    st.markdown("**Vous pouvez saisir manuellement le produit souhaité.** ")
    st.markdown("**Ainsi, une base d'article issue du catalogue Rakuten peut être testée de manière aléatoire en choisissant un type de produit.** ")
    demo_class = st.selectbox("Sélection : ", classes_list) 
    
    if demo_class!='Manuel':
          
        df_st_demo = df[df["prdtypecode"]==int(demo_class[:4])].sample(n=1, random_state=np.random.RandomState())
        
        st.markdown("**Designation** : ")
        st.write(df_st_demo.designation.values[0])
        st.markdown("**Description** : ")
        st.write(df_st_demo.description.values[0])
        
        st.write("**Image de produit** :")
        st.image(Image.open(image_path+df_st_demo.image.values[0]))
        
        
        X_demo_txt = df_st_demo['text_data'].apply(lambda x : " ".join(w for w in x))
        X_demo_txt = tokenizer.texts_to_sequences(X_demo_txt)
        X_demo_txt = tf.keras.preprocessing.sequence.pad_sequences(X_demo_txt, maxlen=max_length, padding='post', truncating='post')
        
        X_demo_img = df_st_demo[['image', 'prdtypecode']] 
        X_demo_img['prdtypecode2']=X_demo_img['prdtypecode'].astype(str)
        demo_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = None)
        demo_set = demo_datagen.flow_from_dataframe(dataframe=X_demo_img,
                                                      directory=image_path,
                                                      x_col = "image",
                                                      y_col = "prdtypecode2",
                                                      class_mode="sparse",
                                                      seed=42,
                                                      target_size = (width, height),
                                                      batch_size = batch,
                                                      shuffle=False)
        
        model_text_W2V = load_model("streamlit_app/assets/text_W2V_Skip-gram.h5")
        demo_pred_txt = model_text_W2V.predict(X_demo_txt)
        demo_pred_class_txt = np.argmax(demo_pred_txt, axis=1)
        demo_pred_class_txt = encoder.inverse_transform(demo_pred_class_txt)
        
        model_image_EfficientNetB1 = load_model("streamlit_app/assets/image_EfficientNetB1.h5")
        demo_pred_img = model_image_EfficientNetB1.predict(demo_set)
        fit_labels = dict((v,k) for k,v in (train_set.class_indices).items())
        demo_pred_class_img = [int(fit_labels[i]) for i in np.argmax(demo_pred_img, axis=1)]  
        
        model_fusion_proba = load_model("streamlit_app/assets/fusion_W2V+EfficientNetB1_2DNN.h5")
        demo_pred_txt_img = np.concatenate([demo_pred_txt, demo_pred_img], axis=1)
        fusion_demo_pred_txt_img = model_fusion_proba.predict(demo_pred_txt_img)
        fusion_demo_pred_txt_img_class = np.argmax(fusion_demo_pred_txt_img, axis=1)
        fusion_demo_pred_txt_img_class = encoder.inverse_transform(fusion_demo_pred_txt_img_class)
  
        
        st.subheader('Résultat : ')
        
        if model_txt :
            st.markdown("**Class prédite (modèle unimodal texte)** :  "+str(demo_pred_class_txt[0])+" - "+classes_dict[demo_pred_class_txt[0]]+ 
                        " ("+str((demo_pred_txt.max()*100).round(2))+" %)")
     
        if model_img :
            st.markdown("**Class prédite (modèle unimodal image)** :  "+str(demo_pred_class_img[0])+" - "+classes_dict[demo_pred_class_img[0]]+
                       " ("+str((demo_pred_img.max()*100).round(2))+" %)")
     
        if model_fus :
            st.markdown("**Class prédite (modèle multimodal)** :  "+str(fusion_demo_pred_txt_img_class[0])+" - "+classes_dict[fusion_demo_pred_txt_img_class[0]]+
                       " ("+str((fusion_demo_pred_txt_img.max()*100).round(2))+" %)")
        
        st.markdown("**Class réelle** :  "+demo_class[:4]+" - "+classes_dict[int(demo_class[:4])])
      
            
    else:   
        
        demo_designation = st.text_input('Designation (obligatoire) :')
     
        demo_description = st.text_input('Description :')
     
        uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'] )
        
        if demo_designation != "" :
 
            manu_demo_txt = word_split(demo_designation + demo_description)
            manu_demo_txt = " ".join(w for w in manu_demo_txt)        
            manu_demo_txt = tokenizer.texts_to_sequences([manu_demo_txt])
            manu_demo_txt = tf.keras.preprocessing.sequence.pad_sequences(manu_demo_txt, maxlen=max_length, padding='post', truncating='post')
     
            model_text_W2V = load_model("streamlit_app/assets/text_W2V_Skip-gram.h5")
            demo_pred_txt = model_text_W2V.predict(manu_demo_txt)
            demo_pred_class_txt = np.argmax(demo_pred_txt, axis=1)
            demo_pred_class_txt = encoder.inverse_transform(demo_pred_class_txt)

        
        if uploaded_file is not None :
            st.image(Image.open(uploaded_file))
            manu_demo_img = image.load_img(uploaded_file, target_size = (width, height))
            manu_demo_img = image.img_to_array(manu_demo_img)
            manu_demo_img = np.expand_dims(manu_demo_img, axis = 0)
        
            model_image_EfficientNetB1 = load_model("streamlit_app/assets/image_EfficientNetB1.h5")
            demo_pred_img = model_image_EfficientNetB1.predict(manu_demo_img)
            fit_labels = dict((v,k) for k,v in (train_set.class_indices).items())
            demo_pred_class_img = [int(fit_labels[i]) for i in np.argmax(demo_pred_img, axis=1)] 
        

            model_fusion_proba = load_model("streamlit_app/assets/fusion_W2V+EfficientNetB1_2DNN.h5")
            demo_pred_txt_img = np.concatenate([demo_pred_txt, demo_pred_img], axis=1)
            fusion_demo_pred_txt_img = model_fusion_proba.predict(demo_pred_txt_img)
            fusion_demo_pred_txt_img_class = np.argmax(fusion_demo_pred_txt_img, axis=1)
            fusion_demo_pred_txt_img_class = encoder.inverse_transform(fusion_demo_pred_txt_img_class)
            

        st.subheader('Résultat : ')

        if model_txt and (demo_designation != ""):
            st.markdown("**Class prédite (modèle unimodal texte)** :  "+str(demo_pred_class_txt[0])+" - "+classes_dict[demo_pred_class_txt[0]]+ 
                        " ("+str((demo_pred_txt.max()*100).round(2))+" %)")
        
        if model_img and (uploaded_file is not None):
            st.markdown("**Class prédite (modèle unimodal image)** :  "+str(demo_pred_class_img[0])+" - "+classes_dict[demo_pred_class_img[0]]+
                       " ("+str((demo_pred_img.max()*100).round(2))+" %)")
        
        if model_fus and (uploaded_file is not None):
            st.markdown("**Class prédite (modèle multimodal)** :  "+str(fusion_demo_pred_txt_img_class[0])+" - "+classes_dict[fusion_demo_pred_txt_img_class[0]]+
                       " ("+str((fusion_demo_pred_txt_img.max()*100).round(2))+" %)")
            
        
