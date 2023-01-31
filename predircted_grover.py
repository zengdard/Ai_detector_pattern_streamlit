import streamlit as st

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

import re

from nltk import ngrams
from collections import defaultdict


import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from nltk.util import ngrams
from nltk.probability import *


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('french'))

st.title('StendhalGPT')
#token = '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'
#API_URL = "https://api-inference.huggingface.co/models/roberta-large-openai-detector"
#headers = {"Authorization": f"Bearer {token}"}

bar = st.progress(0)

st.subheader("Comment fonctionne la richesse lexicale ?")
st.caption(r"La richesse lexicale est cruciale dans la reconnaissance de textes générés par l'IA. Elle se réfère à la variété et à la quantité de mots utilisés dans un texte, qui peuvent influencer la compréhension et l'analyse du contenu par un système informatique. Une richesse lexicale élevée peut aider à améliorer la précision de la reconnaissance de textes générés par l'IA. Cependant, il est important de ne pas confondre richesse lexicale et complexité, car un texte peut être riche en termes de vocabulaire sans pour autant être complexe. En mesurant la richesse lexicale des textes générés par l'IA, nous pouvons améliorer la qualité des analyses et des prédictions effectuées par les modèles. ")

#st.subheader("Comment fonctionne le taux de perplexité ?")
#st.caption(r"")

text = st.text_input("", 'Votre Texte.')

#text_ref = st.text_input("", 'Quel est votre texte de référence ? ')

user_input = text


def grammatical_richness(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words_pos = pos_tag(words)
    words_pos = [word for word in words_pos if word[0] not in stop_words]
    pos = [pos for word, pos in words_pos]
    fdist = FreqDist(pos)
    types = len(fdist.keys())
    tokens = len(words)
    return types / tokens

def verbal_richness(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words_pos = pos_tag(words)
    words_pos = [word for word in words_pos if word[0] not in stop_words]
    verbs = [word for word, pos in words_pos if pos[:2] == 'VB']
    fdist = FreqDist(verbs)
    types = len(fdist.keys())
    tokens = len(words)
    return types / tokens




def lexical_field(text):
    # Tokenization du texte
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("french") + list(string.punctuation))
    words = [word for word in words if word.lower() not in stop_words]
    # Calcul des fréquences des mots
    fdist = FreqDist(words)
    return fdist


def lexical_richness(text):
    # Tokenization du texte
    words = nltk.word_tokenize(text)
    stop_words = set(list(string.punctuation))
    # Calcul de l'étendue du champ lexical
    type_token_ratio = len(set(words)) / len(words)
    return type_token_ratio
    


def highlight_text(text, words_to_highlight):
    # Compile la liste de mots à surligner en une expression régulière
    words_regex = "|".join(words_to_highlight)
    pattern = re.compile(r"\b(" + words_regex + r")\b")
    
    # Remplace les mots correspondant à la condition par des mots entourés de balises HTML <mark>
    highlighted_text = pattern.sub(r"**<span style='background-color: red;'>\1</span>**", text)
    return highlighted_text

model = defaultdict(lambda : defaultdict(lambda : 0.01))
n = 3

def calculate_perplexity(trained_text, test_text):
    trained_tokens = nltk.word_tokenize(trained_text)
    n = len(trained_tokens)
    ngrams = nltk.ngrams(trained_tokens, n)
    model = nltk.NgramModel(n, ngrams)
    test_tokens = nltk.word_tokenize(test_text)
    return model.perplexity(test_tokens)






if st.button('Vérifier via la richesse lexicale.'): 

    st.markdown("### Résultat sur la richesse lexicale.")
    st.caption("La richesse lexicale est un indicateur utilisé pour mesurer la variété, la richesse de mots utilisés dans un texte. Il est calculé en divisant le nombre total de mots uniques dans un texte par le nombre total de mots dans ce même texte. Plus la richesse lexicale est élevée, plus le texte contient de mots différents. Il est important de noter que cet indicateur ne prend pas en compte la pertinence des mots utilisés, seulement leur diversité.")
    st.markdown(f"Taux correspondant à la richesse lexicale de votre texte : **{round(lexical_richness(text),4)}** ")
    st.markdown(f"Taux correspondant à la richesse grammaticale de votre texte : **{round(grammatical_richness(text),4)}** ")
    st.markdown(f"Taux correspondant à la richesse verbale de votre texte : **{round(verbal_richness(text),4)}** ")    
    resul = lexical_field(text)
    try:
        resul2 = resul.most_common(25)   #récupérer les derniers mots car les modèles répètent rarement des sentences.
                            #Donner le nombre de termes pour une meilleur identification. 
    except:
        try:
            resul2 = resul.most_common(5)
        except:
            st.warning('texte trop court.')
    words_to_highlight = []
    for x in resul2:
        words_to_highlight.append(x[0])
    plt.figure(figsize=(10,5))
    resul.plot(30, cumulative=False)
    st.pyplot(plt)
    highlighted_text = highlight_text(text, words_to_highlight)
    st.markdown(highlighted_text, unsafe_allow_html=True)
    bar.progress(100) 



#if st.button('Vérifier via la perplexity.'):

#    perp = calculate_perplexity(text_ref, text)
 #   st.markdown(f"Taux de perplexité de votre texte : **{round(perp,4)}** ")    


