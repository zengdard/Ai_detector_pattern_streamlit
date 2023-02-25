import streamlit as st

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

import re

import PyPDF2
#import docx2txt
from nltk import ngrams
from collections import defaultdict


import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from nltk.util import ngrams
from nltk.probability import *

from math import sqrt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('french'))
from st_on_hover_tabs import on_hover_tabs
st.set_page_config(layout="wide")

st.title('StendhalGPT')
#token = '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'
#API_URL = "https://api-inference.huggingface.co/models/roberta-large-openai-detector"
#headers = {"Authorization": f"Bearer {token}"}

bar = st.progress(0)



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

def highlight_text2(text, words_to_highlight):
    # Compile la liste de mots à surligner en une expression régulière
    words_regex = "|".join(words_to_highlight)
    pattern = re.compile(r"\b(" + words_regex + r")\b")
    
    # Remplace les mots correspondant à la condition par des mots entourés de balises HTML <mark>
    highlighted_text = pattern.sub(r"**<span style='background-color: blue;'>\1</span>**", text)
    return highlighted_text

def distance(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def highlight_text(text, words_to_highlight):
    # Compile la liste de mots à surligner en une expression régulière
    words_regex = "|".join(words_to_highlight)
    pattern = re.compile(r"\b(" + words_regex + r")\b")
    
    # Remplace les mots correspondant à la condition par des mots entourés de balises HTML <mark>
    highlighted_text = pattern.sub(r"**<span style='background-color: red;'>\1</span>**", text)
    return highlighted_text


def calculate_perplexity(trained_text, test_text):
    trained_tokens = nltk.word_tokenize(trained_text)
    n = len(trained_tokens)
    ngrams = nltk.ngrams(trained_tokens, n)
    model = nltk.NgramModel(n, ngrams)
    test_tokens = nltk.word_tokenize(test_text)
    return model.perplexity(test_tokens)

from nltk import bigrams
from collections import Counter

def compare_markov_model(text1, text2):
    # tokenize les deux textes
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)

    # créer des bigrames pour les deux textes
    bigrams1 = list(bigrams(tokens1))
    bigrams2 = list(bigrams(tokens2))

    # compter le nombre d'occurences de chaque bigramme
    count1 = Counter(bigrams1)
    count2 = Counter(bigrams2)

    # mesurer la probabilité de transition pour chaque bigramme dans les deux textes
    prob1 = {bigram: count/len(bigrams1) for bigram, count in count1.items()}
    prob2 = {bigram: count/len(bigrams2) for bigram, count in count2.items()}

    # mesurer la différence entre les deux probabilités pour chaque bigramme
    diff = {bigram: abs(prob1[bigram] - prob2[bigram]) for bigram in prob1.keys() & prob2.keys()}

    return diff

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
with st.sidebar:
        tabs = on_hover_tabs(tabName=['Quésaco ?', 'StendhalGPT'], 
                             iconName=['dashboard', 'money'],
                             styles = {'navtab': {'background-color':'#FFFFFF',
                                                  'color': '#000000',
                                                  'font-size': '18px',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'uppercase'},
                                       'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                      'cursor': 'pointer'}},
                                       'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                       'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'}},
                             key="1")



   


if tabs =='Quésaco ?':
    st.subheader('Quel est le principe d\'utilisation de StendhalGPT ?')

    st.markdown("Nous avons conçu l\'application afin de permettre à ses utilisateurs de justifier ou de mesurer des anomalies dans des textes suspects. Nous appelons 'anomalies' des caractéristiques propores à une génération par une intelligence artificielle ou par des LLMs, et qui caractérisent un textes générés plutôt qu'un texte écrit par un étudiant, c'est à dire un très grand vocabulaire, de très bons taux lexicaux et verbaux, en comparaison avec des copies 'similaires' dans leur exercice de conception.")

    st.subheader('Comment tout cela fonctionne ?') 

    st.markdown("Pour une analyse précise, il est conseillé d'utiliser un ensemble de textes identiques pour les comparer et détecter les anomalies dans les textes générés. Nous travaillons sur des solutions automatiques pour cette comparaison. Si une copie obtient des résultats incohérents, elle sera considérée comme suspecte et pourra être identifiée comme générée si l\'étude des statistiques révèle des résultats trop aléatoires.")

    #st.subheader("Comment fonctionne le taux lexical ?")
    #st.caption(r"La richesse lexicale est cruciale dans la reconnaissance de textes générés par l'IA. Elle se réfère à la variété et à la quantité de mots utilisés dans un texte, qui peuvent influencer la compréhension et l'analyse du contenu par un système informatique. Une richesse lexicale élevée peut aider à améliorer la précision de la reconnaissance de textes générés par l'IA. Cependant, il est important de ne pas confondre richesse lexicale et complexité, car un texte peut être riche en termes de vocabulaire sans pour autant être complexe. En mesurant la richesse lexicale des textes générés par l'IA, nous pouvons améliorer la qualité des analyses et des prédictions effectuées par les modèles. ")
    
    st.subheader("Comment fonctionne la comparaison de deux modèles de Markov ?")
    st.markdown("La comparaison des résultats des modèles de Markov peut aider à déterminer si un texte a été généré par une IA ou non. Une différence significative entre les résultats peut indiquer que le texte a été généré par une IA. En effet, les modèles de Markov mesurent la probabilité de transition entre les mots dans un texte, et les textes générés par une IA ont souvent des transitions différentes de celles des textes réels. Par conséquent, si la différence entre les résultats des modèles est importante, cela peut suggérer que le texte a été généré par une IA plutôt que par un être humain.")
        

elif tabs == 'StendhalGPT':

    with col3:

        text_ref = st.text_input("Insérez un/vos texte(s) référent dans cette colonne.", '')
        pdf_file2 = st.file_uploader("Télécharger plusieurs textes de référence au format PDF", type="pdf", accept_multiple_files=True)

        if pdf_file2 is not None:
            for pdf_fil2 in pdf_file2:
                # Lecture du texte de chaque fichier PDF
                pdf_reader = PyPDF2.PdfReader(pdf_fil2)
                for page in range(len(pdf_reader.pages)):
                    text_ref += pdf_reader.pages[page].extract_text()

        # Ajout du widget pour télécharger plusieurs fichiers DOCX
        #docx_files = st.file_uploader("Télécharger plusieurs textes de référence au format DOCX", type="docx", accept_multiple_files=True)

        #if docx_files is not None:
         #   for docx_file in docx_files:
                # Lecture du texte de chaque fichier DOCX
          #      text_ref += docx2txt.process(docx_file)

    
    with col4:
        text = st.text_input("Insérez un/vos textes suspects/ à comparaître dans cette colonne", '')
        pdf_file = st.file_uploader("Télécharger plusieurs textes à comparer au format PDF", type="pdf", accept_multiple_files=True)
       # docx_file2s = st.file_uploader("Télécharger plusieurs textes à comparer au format DOCX", type="docx", accept_multiple_files=True)

        #if docx_file2s is not None:
         #   for docx_file2 in docx_file2s:
                # Lecture du texte de chaque fichier DOCX
          #      text += docx2txt.process(docx_file2)


        if pdf_file is not None:
            for pdf_fil in pdf_file:
            # Lecture du texte de chaque fichier PDF
                pdf_reader = PyPDF2.PdfReader(pdf_fil)
                
                for page in range(len(pdf_reader.pages)):
                        text += pdf_reader.pages[page].extract_text()
        

    if st.button('Vérifier via le taux lexical.'): #intégrer le texte de référence pour plus de rapidité 

        with col1:

            richesse_lex = round(lexical_richness(text),4)
            richesse_gram = round(grammatical_richness(text),4)
            richesse_verbale = round(verbal_richness(text),4)

            st.markdown("### Résultat sur le taux moyen lexical pour le(s) texte(s) à comparaître.")
            st.markdown(f"Taux correspondant au taux lexical de votre texte(s) : **{richesse_lex}** ")
            st.markdown(f"Taux correspondant au taux grammatical de votre texte(s) : **{richesse_gram}** ")
            st.markdown(f"Taux correspondant au taux verbal de votre texte(s) : **{richesse_verbale}** ")    
            resul = lexical_field(text)

            try:
                resul2 = resul.B()
                resul2 = resul.most_common(resul2)
                print(resul2)

                                    #Demander le nombre de termes pour une meilleur identification. 
            except:
                try:
                    resul2 = resul.B()
                    print(resul2)
                    resul2 = resul.most_common(resul2)

                except:
                    st.warning('Texte trop court.')
            words_to_highlight = []
            i = 34
            df = pd.DataFrame(resul2, columns=['Mots', 'Occurence'])

            st.dataframe(df)
            #for x in resul2 : 
            #   words_to_highlight.append(x[0])
            #  bar.progress(i + 2)
            #words_to_highlight_reverse = list(reversed(words_to_highlight))
            
            plt.figure(figsize=(10,5))
            resul.plot(30, cumulative=False)
            bar.progress(79) 
            st.pyplot(plt)
        # xslider = st.slider('Sélectionner une valeur', min_value=0, max_value=len(resul2), value=4)
        # highlighted_text = highlight_text(text, words_to_highlight[:2])
            #highlighted_text2 = highlight_text2(text, words_to_highlight_reverse[2:])
            #st.markdown("**Mots les moins cités** : \n" +str(highlighted_text2), unsafe_allow_html=True)
            #st.markdown("**Mots les plus cités** : \n" +str(highlighted_text), unsafe_allow_html=True)
            
        ##Pour le texte de Référence 
        with col2 :
            text = text_ref

            richesse_lex2 = round(lexical_richness(text),4)
            richesse_gram2 = round(grammatical_richness(text),4)
            richesse_verbale2 = round(verbal_richness(text),4)

            st.markdown("### Résultat sur le taux moyen lexical pour le(s) texte(s) de référence.")
            st.markdown(f"Taux correspondant au taux lexical de votre texte(s) : **{richesse_lex2}** ")
            st.markdown(f"Taux correspondant au taux grammatical de votre texte(s) : **{richesse_gram2}** ")
            st.markdown(f"Taux correspondant au taux verbal de votre texte(s) : **{richesse_verbale2}** ")    

            resul = lexical_field(text)
            
            try:
                resul2 = resul.B()
                resul2 = resul.most_common(resul2)
                

                                    #Demander le nombre de termes pour une meilleur identification. 
            except:
                try:
                    resul2 = resul.B()
                    resul2 = resul.most_common(resul2)
                    print(resul2)

                except:
                    st.warning('Texte trop court.')
            words_to_highlight = []
            i = 34
        # for x in resul2 : 
            #    words_to_highlight.append(x[0])
            #   bar.progress(i + 2)
            #words_to_highlight_reverse = list(reversed(words_to_highlight))
            df = pd.DataFrame(resul2, columns=['Mots', 'Occurence'])

            st.dataframe(df)
            plt.figure(figsize=(10,5))
            resul.plot(30, cumulative=False)
            bar.progress(79) 
            st.pyplot(plt)

        # yslider = st.slider('Sélectionner une valeur', min_value=0, max_value=len(resul2), value=4,)  #Doit quand modifier remettre à jour le texte surligné

            #highlighted_text = highlight_text(text, words_to_highlight[:4])
            #highlighted_text2 = highlight_text2(text, words_to_highlight_reverse[4:])
            #st.markdown("**Mots les moins cités** : \n" +str(highlighted_text2), unsafe_allow_html=True)
            #st.markdown("**Mots les plus cités** : \n" +str(highlighted_text), unsafe_allow_html=True)
            

        # Liste des coordonnées x et y des points à afficher

        x = [richesse_lex, richesse_lex2]
        y = [richesse_gram/richesse_verbale, richesse_gram2/richesse_verbale2]

        # Création du graphique
        fig, ax = plt.subplots()
        ax.scatter(x, y)

        # Ajout d'étiquettes et de limites aux axes
        ax.set_xlabel("Taux lexical")
        ax.set_ylabel("Rapport taux grammatical sur verbal")
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 12])

        st.pyplot(plt)

        dist = distance((x[0], y[0]), (x[1],y[1]))
        st.markdown(f"Distance entre les points {dist}.")

        bar.progress(100) 


    if st.button('Vérifier via le modèle de Markov.'):
        resultat  = compare_markov_model(text, text_ref)
        key_list = []
        value_list = []
        for key, value in resultat.items():
            key_list.append(key)
            value_list.append(value)
        df = pd.DataFrame(list(zip(key_list,value_list)), columns = ['Texte','Probabilité'])

    #convertion du dictionnaire en dataframe

        st.dataframe(df)
