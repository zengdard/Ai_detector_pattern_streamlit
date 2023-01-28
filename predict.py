import requests
import streamlit as st

import nltk
#nltk.download('popular')
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('french'))

st.title('StendhalGPT')
token = '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'
API_URL = "https://api-inference.huggingface.co/models/roberta-large-openai-detector"
headers = {"Authorization": f"Bearer {token}"}
st.text('Cet outil est en version de développement.' )

st.subheader("Fonctionnement via GPT-2.")
st.text('Cette option est valable uniquement pour les textes en Anglais et fonctionne avec le modèle GPT-2 Detector.' )

st.subheader("Fonctionnement via la richesse lexicale.")
st.text('Cette option est valable pour tous les textes pour toutes les langues.')

def query(payload):
    
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

text = st.text_input("", 'Votre Texte.')
user_input = text
'''
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
'''



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
    
bar = st.progress(0)

if st.button('Vérifier via GPT-2'):

    output = query({
	"inputs": user_input,
        })
    bar.progress(20)
    try:
        st.subheader("Résultat via GPT-2")
        if output[0][1]['score'] >=  output[0][0]['score']:
            st.text('Votre texte est à'+ str(round(output[0][0]['score'], 2)*100)+'% de chance d\'être génèré par une IA.')    
        else : 
            st.text('Votre texte est à '+ str(round(output[0][1]['score'], 2)*100)+'% de chance d\'être écrit par un humain.' )
    except:
        st.warning('Le service est temporairement indisponible pour GPT-2, merci de votre compréhension.' )
    bar.progress(100) 

if st.button('Vérifier via la richesse lexicale'): 
    st.markdown("### Résultat sur la richesse lexicale.")
    st.st.text("La richesse lexicale est un indicateur utilisé en traitement automatique du langage naturel (NLP) pour mesurer la variété de mots utilisés dans un texte. Il peut être calculé en divisant le nombre total de mots uniques dans un texte par le nombre total de mots dans ce même texte. Plus la richesse lexicale est élevée, plus le texte contient de mots différents. Il est important de noter que cet indicateur ne prend pas en compte la pertinence des mots utilisés, seulement leur diversité. Il est souvent utilisé pour évaluer la qualité de la langue d'un texte")
    st.markdown(f"Taux correspondant à la richesse lexicale de votre texte : {lexical_richness(text)} ")
    #st.markdown(f"Taux correspondant à la richesse grammaticale de votre texte : {grammatical_richness(text)} ")
    #st.markdown(f"Taux correspondant à la richesse verbale de votre texte : {verbal_richness(text)} ")    
    resul = lexical_field(text)
    resul = resul.most_common(4)
 #   detail = [('Richesse lexicale',lexical_richness(text)),('Richesses Grammaticale',grammatical_richness(text)), ('Richesse Verbale',verbal_richness(text))]
    #df = pd.DataFrame(resul, columns=["Mots", "Fréquences"])
#    df_2 = pd.DataFrame(detail, columns=["Type", "Taux"])
    #st.dataframe(df)
    bar.progress(100) 
