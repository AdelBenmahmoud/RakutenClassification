import streamlit as st
from PIL import Image


title = "Classification de produits e-commerce Rakuten"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")
    
    st.markdown(
        """
        ## Contexte
        """
    )
    
    st.image(Image.open("assets/logo-rakuten.png"))

    st.markdown(
        """
        Rakuten, créé en 1997 au Japon et à l'origine du concept de marketplace, est devenu l'une des plus grandes plateformes de e-commerce au monde avec une communauté de plus de 1,3 milliard de membres.
        
        Rakuten Institute of Technology (RIT) est le département de recherche et d'innovation de Rakuten. RIT fait de la recherche appliquée dans les domaines de la vision par ordinateur, du traitement du langage naturel, de l'apprentissage machine/profond et de l'interaction homme-machine. Le 6 janvier 2020, RIT propose le challenge suivant : [Rakuten France Multimodal Product Data Classification](https://challengedata.ens.fr/challenges/35).
        
        Ce défi porte sur le thème de la classification multimodale (texte et image) des codes types de produits à grande échelle où l'objectif est de prédire le code type de chaque produit tel que défini dans le catalogue de Rakuten France.
        
        Le catalogage des produits via la catégorisation des titres et des images est un problème fondamental pour tout marché de e-commerce, avec des applications allant de la recherche et des recommandations personnalisées à la compréhension des requêtes. Les approches manuelles basées sur des règles de catégorisation ne sont pas évolutives puisque les produits commerciaux sont organisés en plusieurs classes. Le déploiement d'approches multimodales serait une technique utile pour les entreprises de e-commerce car elles ont du mal à catégoriser les produits en fonction des images et des labels des marchands et à éviter les doublons, en particulier lors de la vente de produits neufs et d'occasion de marchands professionnels et non professionnels.
        
        Les progrès dans ce domaine de recherche ont été limités par le manque de données réelles provenant de catalogues commerciaux réels. Ce projet présente plusieurs aspects de recherche intéressants en raison de la nature intrinsèquement bruyante des labels et des images des produits, de la taille des catalogues de e-commerce et de la distribution déséquilibrée typique des données.
        
        ## Description du problème
        L'objectif de ce défi des données est la classification à grande échelle des données produit multimodales (texte et image) en codes de type de produit.
        
        Par exemple, dans le catalogue Rakuten France, un produit avec une désignation ou un titre français "Klarstein Présentoir 2 Montres Optique Fibre" associé à une image et parfois à une description complémentaire. Ce produit est classé sous le code de type de produit 1500. Il existe d'autres produits avec des titres, des images et des descriptions possibles différents, qui se trouvent sous le même code de type de produit. Compte tenu de ces informations sur les produits, comme l'exemple ci-dessus, ce défi propose de modéliser un classificateur pour classer les produits dans son code de type de produit correspondant.
        
        ## Description des données
        Pour ce challenge, Rakuten France met à disposition env. 99 000 listes de produits au format CSV, y compris le train (84 916) et l'ensemble de test (13 812). L'ensemble de données se compose de désignations de produits, de descriptions de produits, d'images de produits et de leur code de type de produit correspondant.
        
        Les données sont réparties selon deux critères, formant quatre ensembles distincts : entraînement ou test, entrée ou sortie.
        - X_train.csv : fichier d'entrée d'entraînement
        - Y_train.csv : fichier de sortie d'entraînement
        - X_test.csv : fichier d'entrée de test

        De plus, le fichier images.zip est fourni contenant toutes les images. La décompression de ce fichier fournira un dossier nommé images avec deux sous-dossiers nommés image_train et image_test, contenant respectivement des images d'entraînement et de test.
        
        """
    )
