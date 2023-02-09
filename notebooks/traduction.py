# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
import numpy as np
import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('french'))

from googletrans import Translator
translator = Translator()

def word_translate(words):
    """
    translate all in french and remove stopwords
    input words : a list of words
    
    """            
    #translate all in french
    words_trans=[]
    translations = translator.translate(words, dest='fr')
    for word in translations:
        words_trans.append(word.text.split(" "))
    words_trans_flatten = [item for sublist in words_trans for item in sublist]
        
    # convert to lower case
    words_trans_flatten = [w.lower() for w in words_trans_flatten]
    
    # filter out stop words    
    words_final = [w for w in words_trans_flatten if not w in stop_words]
    
    return words_final


words = pd.read_csv('designation-description.csv', index_col=0, squeeze=True)
#words = words.loc[83911:]

table = str.maketrans('', '', string.punctuation)

with open('traduction_designation-description.txt', 'a', encoding='utf-8') as f:
    for w, index in zip(words.values, words.index):
        w = w.translate(table).split(' ')
        w_trans = word_translate(w)
        f.write(str(w_trans)+'\n')
        print(index)
        #print(w_trans)

