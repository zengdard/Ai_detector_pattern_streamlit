import streamlit as st

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

import re

import statistics


from nltk import bigrams
from collections import Counter

import math

import language_tool_python


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
    stop_words = set(stopwords.words("french") + list(string.punctuation))

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


def count_words(text):
    words = text.split()
    return len(words)

def compute_text_metrics(lexical_density,grammatical_density,verbal_density):
 
    # Ajuster les pondérations pour donner plus de poids au taux verbal
    lexical_weight = 0.25
    grammatical_weight = 0.55
    verbal_weight = 1.5
    
    # Calculer la mesure agrégée en combinant les trois taux avec des pondérations ajustées
    text_metric = (lexical_weight * lexical_density) + (grammatical_weight * grammatical_density) + (verbal_weight * verbal_density)
    ### Plus proche de 1 alors texte très riche et donc suspect 


    return text_metric 


import statistics

def measure_lexical_richness(text, punctuation):
    # Diviser le texte en phrases
    sentences = nltk.sent_tokenize(text.lower())
    
    # Diviser chaque phrase en blocs en utilisant la ponctuation spécifiée
    blocks = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        sentence_blocks = []
        current_block = ''
        for word in words:
            if word in punctuation:
                if current_block:
                    sentence_blocks.append(current_block)
                sentence_blocks.append(word)
                current_block = ''
            else:
                current_block += ' ' + word
        if current_block:
            sentence_blocks.append(current_block.strip())
        blocks.append(sentence_blocks)
    
    # Calculer le taux lexical de chaque bloc
    lexical_densities = [len(set(block))/len(block) for sentence in blocks for block in sentence if len(block) > 0]
    
    # Calculer la moyenne et l'écart-type des taux lexicaux
    avg_ld = sum(lexical_densities)/len(lexical_densities)
    std_ld = statistics.stdev(lexical_densities)
    
    # Trouver les blocs anormaux (ceux dont le taux lexical est en dehors de deux écarts-types de la moyenne)
    abnormal_blocks = [block for sentence in blocks for block in sentence if len(block) > 0 and (len(set(block))/len(block) > (avg_ld + 2*std_ld) or len(set(block))/len(block) < (avg_ld - 2*std_ld))]
    
    return abnormal_blocks 



def spell_check(text):
    tool = language_tool_python.LanguageTool('fr')
    matches = tool.check(text)
    corrected_text = language_tool_python.correct(text, matches)
    return corrected_text



def detect_generated_text(text):
    # Tokenize le texte en mots
    words = text.lower().split()
    
    # Calculer la fréquence de chaque mot
    word_freq = {}
    for word in words:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1
    
    # Calculer le nombre de mots uniques
    num_unique_words = len(word_freq)
    
    # Calculer la proportion de mots uniques dans le texte
    unique_word_ratio = num_unique_words / len(words)
    
    # Calculer le poids de chaque mot en fonction de sa fréquence et de la diversité de mots dans le texte
    word_weights = {}
    for word in word_freq:
        # Le poids de chaque mot est calculé en multipliant la fréquence du mot par un facteur de normalisation basé sur la diversité de mots dans le texte
        weight = word_freq[word] * (1 - (math.log(num_unique_words) / math.log(len(words))))
        word_weights[word] = weight
        
    # Calculer le poids total du texte en faisant la somme des poids de chaque mot
    total_weight = sum(word_weights.values())
    
    # Appliquer une formule pour donner un poids plus important aux textes ayant une proportion élevée de mots uniques
    if unique_word_ratio >= 0.4:
        score = (unique_word_ratio - 0.4) / 0.6
    else:
        score = 0
        
    # Ajouter le score basé sur la proportion de mots uniques au poids total du texte
    weighted_score = total_weight + score
    
    return weighted_score






def count_characters(text):
    return len(text)

col5, col6 = st.columns(2)
col3, col4 = st.columns(2)
col1, col2 = st.columns(2)


with st.sidebar:
        tabs = on_hover_tabs(tabName=['Quésaco ?','StendhalGPT', 'StendhalGPT Expert'], 
                             iconName=['dashboard','description',  'toll'],
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

    st.markdown("Nous avons conçu l\'application afin de permettre à ses utilisateurs de justifier ou de mesurer des anomalies dans des textes suspects. Nous appelons 'anomalies' des caractéristiques propres à une génération par une intelligence artificielle ou par des LLMs, et qui caractérisent un textes générés plutôt qu'un texte écrit par un étudiant, c'est à dire un très grand vocabulaire, de très bons taux lexicaux et verbaux, en comparaison avec des copies 'similaires' dans leur exercice de conception.")

    st.subheader('Comment tout cela fonctionne ?') 

    st.markdown("Pour une analyse précise, il est conseillé d'utiliser un ensemble de textes identiques pour les comparer et détecter les anomalies dans les textes générés. Nous travaillons sur des solutions automatiques pour cette comparaison. Si une copie obtient des résultats incohérents, elle sera considérée comme suspecte et pourra être identifiée comme générée si l\'étude des statistiques révèle des résultats trop aléatoires.")

    #st.subheader("Comment fonctionne le taux lexical ?")
    #st.caption(r"La richesse lexicale est cruciale dans la reconnaissance de textes générés par l'IA. Elle se réfère à la variété et à la quantité de mots utilisés dans un texte, qui peuvent influencer la compréhension et l'analyse du contenu par un système informatique. Une richesse lexicale élevée peut aider à améliorer la précision de la reconnaissance de textes générés par l'IA. Cependant, il est important de ne pas confondre richesse lexicale et complexité, car un texte peut être riche en termes de vocabulaire sans pour autant être complexe. En mesurant la richesse lexicale des textes générés par l'IA, nous pouvons améliorer la qualité des analyses et des prédictions effectuées par les modèles. ")
        
    st.subheader('L\'indicateur de richesse générale') 

    st.markdown('L\'indicateur de richesse lexcicale est le résultat de la mesure agrégée entre la richesse lexicale, verbale, et grammaticale avec des pondérations ajustées afin de mesurer plus justement la richesse lexicale du\'texte. Plus cet indicateur est élevé plus celui-ci est riche. ')

    st.subheader('L\'indicateur de richesse modale') 

    st.markdown('L\'indicateur de richesse modale mesure la richesse d\'un texte en fonction de la fréquence d\'apparitions des mots dans un corpus en donnant plus de poids aux textes contenant beaucoup de mots peu utilisés. Plus un texte contient une grande diversité de mots avec une faible fréquence plus celui-ci aura un poids beaucoup plus important.')

    st.subheader('La différence relative')

    st.markdown('La différence relative permet de mettre en évidence les différences significatives entre deux textes. Plus cet indicateur est élevé plus celui-ci indique que les textes sont significativement différents les uns des autres. ')


   ## st.subheader("Comment fonctionne la comparaison de deux modèles de Markov ?")
   # st.markdown("La comparaison des résultats des modèles de Markov peut aider à déterminer si un texte a été généré par une IA ou non. Une différence significative entre les résultats peut indiquer que le texte a été généré par une IA. En effet, les modèles de Markov mesurent la probabilité de transition entre les mots dans un texte, et les textes générés par une IA ont souvent des transitions différentes de celles des textes réels. Par conséquent, si la différence entre les résultats des modèles est importante, cela peut suggérer que le texte a été généré par une IA plutôt que par un être humain.")
        
elif tabs == 'StendhalGPT':
    
    with col5:

        text = st.text_input("Insérez un/vos texte(s) référent dans cette colonne.", '')
        pdf_file2 = st.file_uploader("Télécharger plusieurs textes de référence au format PDF", type="pdf", accept_multiple_files=True)

        if pdf_file2 is not None:
            for pdf_fil2 in pdf_file2:
                # Lecture du texte de chaque fichier PDF
                pdf_reader = PyPDF2.PdfReader(pdf_fil2)
                for page in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page].extract_text()

        # Ajout du widget pour télécharger plusieurs fichiers DOCX
        #docx_files = st.file_uploader("Télécharger plusieurs textes de référence au format DOCX", type="docx", accept_multiple_files=True)

        #if docx_files is not None:
         #   for docx_file in docx_files:
                # Lecture du texte de chaque fichier DOCX
          #      text_ref += docx2txt.process(docx_file)

    
    with col6:
        
        text_ref = st.text_input("Insérez un/vos textes suspects/ à comparaître dans cette colonne", '')
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
                        text_ref += pdf_reader.pages[page].extract_text()
        

    if st.button('Vérifier.'):
        try : 

            texte_metrique2 = detect_generated_text(text_ref)
            texte_metrique = detect_generated_text(text)

            resulabs = abs(texte_metrique2 - texte_metrique) /  texte_metrique2
            print(resulabs)
            
            st.markdown(f'## La différence relative entre vos textes est de :red[{round(resulabs,4)*100}]')

        except:
            st.warning('Un de vos textes est trop court.')


elif tabs == 'StendhalGPT Expert':

    with col3:

        text = st.text_input("Insérez un/vos texte(s) référent dans cette colonne.", '')
        pdf_file2 = st.file_uploader("Télécharger plusieurs textes de référence au format PDF", type="pdf", accept_multiple_files=True)

        if pdf_file2 is not None:
            for pdf_fil2 in pdf_file2:
                # Lecture du texte de chaque fichier PDF
                pdf_reader = PyPDF2.PdfReader(pdf_fil2)
                for page in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page].extract_text()

        # Ajout du widget pour télécharger plusieurs fichiers DOCX
        #docx_files = st.file_uploader("Télécharger plusieurs textes de référence au format DOCX", type="docx", accept_multiple_files=True)

        #if docx_files is not None:
         #   for docx_file in docx_files:
                # Lecture du texte de chaque fichier DOCX
          #      text_ref += docx2txt.process(docx_file)

    
    with col4:
        
        text_ref = st.text_input("Insérez un/vos textes suspects/ à comparaître dans cette colonne", '')
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
                        text_ref += pdf_reader.pages[page].extract_text()
        

    if st.button('Vérification via le taux lexical.'): #intégrer le texte de référence pour plus de rapidité 
        if text == '' or text_ref=='':
            st.warning("Veuillez indiquer votre texte.")
        else:
                with col1:


                    try :
                        richesse_gram = round(grammatical_richness(text),4)
                        richesse_detect = detect_generated_text(text)

                        richesse_lex = round(lexical_richness(text),4)
                        richesse_verbale = round(verbal_richness(text),4)


                        compteur_mots = count_words(text)
                        texte_metrique = round(compute_text_metrics(richesse_lex,richesse_gram,richesse_verbale ),4)

                        nrb = count_characters(text)

                        st.markdown("### Résultat sur le taux moyen lexical pour le(s) texte(s) référent.")
                        st.markdown(f"Taux correspondant au taux lexical de votre texte(s) : **{richesse_lex}** ")
                        st.markdown(f"Taux correspondant au taux grammatical de votre texte(s) : **{richesse_gram}** ")
                        st.markdown(f"Taux correspondant au taux verbal de votre texte(s) : **{richesse_verbale}** ") 
                        st.markdown(f"Nombre de mots : **{compteur_mots}** ")  

                        st.markdown(f"Nombre de caractères: **{nrb}** ")  

                        st.markdown(f"### Indicateur de richesse générale :  :red[{round(texte_metrique * 100,4)}]")  
                        st.markdown(f"### Indicateur de richesse modale :  :red[{round(richesse_detect * 100,4)}]")  
                    except:
                        st.warning("Votre texte est trop court.")
                        


                    ##block étrange
                    #block = measure_lexical_richness(text, 5)


                    #Fréquence des mots 
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
                    
                

                    try :

                            
                        richesse_lex2 = round(lexical_richness(text_ref),4)
                        richesse_gram2 = round(grammatical_richness(text_ref),4)
                        richesse_verbale2 = round(verbal_richness(text_ref),4)
                        texte_metrique2 = round(compute_text_metrics(richesse_lex2,richesse_gram2,richesse_verbale2 ),4)

                        compteur_mots = count_words(text_ref)
                        nrb = count_characters(text_ref)
                        
                        
                        richesse_detect2 = detect_generated_text(text_ref)

                            
                        st.markdown("### Résultat sur le taux moyen lexical pour le(s) texte(s) à comparaître.")
                        st.markdown(f"Taux correspondant au taux lexical de votre texte(s) : **{richesse_lex2}** ")
                        st.markdown(f"Taux correspondant au taux grammatical de votre texte(s) : **{richesse_gram2}** ")
                        st.markdown(f"Taux correspondant au taux verbal de votre texte(s) : **{richesse_verbale2}** ") 
                        st.markdown(f"Nombre de mots : **{compteur_mots}** ")   


                        st.markdown(f"Nombre de caractères: **{nrb}** ")  

                        st.markdown(f"### Indicateur de richesse :  :red[{round(texte_metrique2 * 100,4)}]")  
                        st.markdown(f"### Indicateur de richesse modale :  :red[{round(richesse_detect2 * 100,4)}]") 

                    except:
                        st.warning("Votre texte est trop court.")
                     



                    resul = lexical_field(text_ref)
                    
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
                                st.warning('Votre texte est trop court.')

                    i = 34
              


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

                try : 
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
                except:
                    st.info('Textes trop courts pour une représentation 2D.')

                bar.progress(100) 
            



                #if st.button('Vérifier via le modèle de Markov.'):
                 #   resultat  = compare_markov_model(text, text_ref)
                 #   key_list = []
                 #   value_list = []
                 #   for key, value in resultat.items():
                 #       key_list.append(key)
                 #       value_list.append(value)
                 #   df = pd.DataFrame(list(zip(key_list,value_list)), columns = ['Texte','Probabilité'])

                #convertion du dictionnaire en dataframe

                  #  st.dataframe(df)

#
#elif tabs == "StendhalGPT FusionedText":
#    
#    st.subheader("StendhalGPT FusionedText")
#    st.markdown("StendhalGPT tente d'identifier les parties du texte qui peuvent être générées et intégrées dans le corpus en repérant des données statistiques anormales par rapport à tout le texte complet. Les phrases surlignées en rouge, possèdent des statistiques en dehors de l'écart type de la moyenne de tout le texte.")
#    st.info('StendhalGPT FusionedText est susceptible d\'évoluer.')
#    text = st.text_input("Entrez votre texte ici.")
#    ponctu = st.radio(
#        "Sélectionnez une ponctuation pour partitionner votre texte",
#        key="visibility",
#        options=["Espace", "Point", "Virgule"],
#    )
#    if ponctu == 'Espace':
#        ponctu = ' '
#    elif ponctu == 'Point':
#        ponctu = '.'
#    else :
#        ponctu = ","
#
#        
#    if st.button('Vérifier'):
#
#        anormal_bloc = measure_lexical_richness(text, ponctu)
#        
#            
#        highlighted_text = highlight_text(text, [word for block in anormal_bloc for word in block])
#
#        st.markdown("**Blocs Anormaux** : \n" +str(highlighted_text), unsafe_allow_html=True)
#
#
#
