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


import openai
openai.api_key = "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"

import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from nltk.probability import *

from math import sqrt

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('french'))
from st_on_hover_tabs import on_hover_tabs
st.set_page_config(layout="wide")

st.title('StendhalGPT')


bar = st.progress(0)

def generation2(thm):
    result = ''
    bar.progress(32)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": f"écris moi uniquement un texte suivant ces paramètres {thm}"},
            ]
    )
    bar.progress(89)
    for choice in response.choices:
        result += choice.message.content + '\n'
    return result

def generation(thm):
    
    bar.progress(32)
    openai.api_key = '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'
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=thm,
    max_tokens=2048,
    temperature=0
        )
    bar.progress(80)
    answer = response.choices[0].text
    return answer


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
    global prob1
    global prob2
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


    common_bigrams = set(count1.keys()) & set(count2.keys())
    # Obtenir les probabilités pour chaque bigramme commun
    prob1 = {bigram: count1[bigram] / sum(count1.values()) for bigram in common_bigrams}
    prob2 = {bigram: count2[bigram] / sum(count2.values()) for bigram in common_bigrams}
    
    
    # mesurer la différence entre les deux probabilités pour chaque bigramme
    #diff = {bigram: abs(prob1[bigram] - prob2[bigram]) for bigram in prob1.keys() & prob2.keys()}

    return [prob1, prob2]



def count_words(text):
    words = text.split()
    return len(words)

def compute_text_metrics(lexical_density,grammatical_density,verbal_density):
 
    # Ajuster les pondérations pour donner plus de poids au taux verbal
    lexical_weight = 0.25
    grammatical_weight = 0.55
    verbal_weight = 1.75
    
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
import string

def lexical_richness_normalized(text1, text2):
    # Tokenization
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)
    
    # Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens1 = [w.translate(table) for w in tokens1]
    tokens2 = [w.translate(table) for w in tokens2]
    
    # Number of unique words
    unique1 = len(set(tokens1))
    unique2 = len(set(tokens2))
    
    # Total number of words
    total1 = len(tokens1)
    total2 = len(tokens2)
    
    # Type-token ratio
    ttr1 = unique1 / total1
    ttr2 = unique2 / total2
    
    # Measure of Textual Lexical Diversity
    mtd1 = len(set(tokens1)) / len(tokens1)
    mtd2 = len(set(tokens2)) / len(tokens2)
    
    # Return normalized values in a dictionary
    return {'unique_words_ratio': (unique1 / unique2) if unique2 > 0 else 0,
            'total_words_ratio': (total1 / total2) if total2 > 0 else 0,
            'ttr_ratio': ttr1 / ttr2 if ttr2 > 0 else 0,
            'mtd_ratio': mtd1 / mtd2 if mtd2 > 0 else 0}





def count_characters(text):
    return len(text)

col5, col6 = st.columns(2)
col3, col4 = st.columns(2)
col1, col2 = st.columns(2)


with st.sidebar:
        tabs = on_hover_tabs(tabName=['Quésaco ?','StendhalGPT', 'StendhalGPT Expert', 'StendhalGPT MultipleTextes'], 
                             iconName=['help center','home',  'toll', 'analytics'],
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

def calculate_stats(texts):
    # Séparer les textes en une liste
    

    # Supprimer les éléments vides
    texts = list(filter(None, texts))
    num_texts = []
    avg_length = []
    # Calculer le nombre de textes
    for x in texts : 
        num_texts.append(verbal_richness(x))
        avg_length.append(compute_text_metrics(lexical_richness(x),grammatical_richness(x),verbal_richness(x)))
    # Créer un DataFrame pandas pour stocker les résultats
    stats_df = pd.DataFrame({
        "Nombre": num_texts,##Verbal
        "Longueur": avg_length##Combiné et pondération
    })

    colors = plt.cm.Set1(np.linspace(0, 1, len(num_texts)))


    fig, ax = plt.subplots()
    for i in range(0, len(num_texts)):
        ax.scatter(x=num_texts[i], y=avg_length[i], c=[colors[i]], label=f"Texte {i+1}")
    ax.scatter(x=np.mean(num_texts), y=np.mean(avg_length), c="black", marker=".", s=300, label="Moyenne")
    max_num_texts = max(num_texts)
    min_num_texts = min(num_texts)
    max_avg_length = max(avg_length)
    min_avg_length = min(avg_length)

    # Définir les limites du graph en utilisant les valeurs maximales et minimales
    ax.set_xlim(min_num_texts - 0.1, max_num_texts + 0.1)
    ax.set_ylim(min_avg_length - 0.1, max_avg_length + 0.1)


    ax.set_xlabel("Taille du champ verbal")
    ax.set_ylabel("Taille de la richesse générale")
    ax.set_title(f"Visualisation de la richesse lexicale par rapport à la richesse générale.")
    ax.legend()
    st.pyplot(fig)



def kl_div_with_exponential_transform(mat1, mat2, alpha):
    # Transform matrices
    mat1_transformed = np.exp(alpha * mat1)
    mat2_transformed = np.exp(alpha * mat2)
    
    # Normalize matrices
    mat1_normalized = mat1_transformed / np.sum(mat1_transformed, keepdims=True)
    mat2_normalized = mat2_transformed / np.sum(mat2_transformed, keepdims=True)
    
    # Calculate KL divergence
    kl_div = np.sum(mat1_normalized * np.log(mat1_normalized / mat2_normalized))
    
    return kl_div


def shifted_exp(x, shift=0):
    return 1+(1 / ((x - shift)**10))




def scaled_manhattan_distance(a, b):
    scale_factor = sum(a.shape) + sum(b.shape)
    return np.sum(np.abs(a - b)) / scale_factor

def is_within_10_percent(x, y):
    threshold = 0.2  # 20%
    difference = abs(x - y)
    avg = (x + y) / 2
    return difference <= (avg * threshold)


def create_markdown_table(similarity_measures):
    table_header = '| Mesure | Taux de similarité |\n'
    table_divider = '| ------- | ---------- |\n'
    table_rows = ''
    for measure, similarity in similarity_measures.items():
        table_rows += f'| {measure} | {similarity} |\n'
    markdown_table = table_header + table_divider + table_rows
    return markdown_table

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

    st.info('En dessous de 130 mots, il est préférable d\'utiliser la fonction Expert. ')
    
    with col5:

        text = st.text_input("Insérez un/vos texte(s) référent dans cette colonne.", '')
       
    
    with col6:
        nbr_mots_text = len(text.split(" "))
        #print(nbr_mots_text)
        bar.progress(0)
        text_ref = st.text_input("Insérez un descriptif de votre texte (taille, type, sujet, niveau d'études.)")
        try:
            
            text_ref = generation2('Génére uniquement un text en français en respectant ces critères : '+text_ref+' en '+str(nbr_mots_text)+'nombre de mots')
        except:
            try:
                text_ref = generation('Génére uniquement un text en français en respectant ces critères : '+text_ref+' en '+str(nbr_mots_text)+'nombre de mots')
            except:
                st.warning('Le service est surchargé, veuiller utiliser une autre méthode.')


    if st.button('Vérifier simplement'):
            
        try : 
            diff = compare_markov_model(text, text_ref)
            vec1 = np.array([diff[0][bigram] for bigram in prob1] +[verbal_richness(text)]+[grammatical_richness(text)]+[lexical_richness(text)] )
            vec2 = np.array([diff[1][bigram] for bigram in prob2] +[verbal_richness(text_ref)]+[grammatical_richness(text_ref)]+[lexical_richness(text_ref)])
                        
            x = len(vec1)

            moye = shifted_exp(x)
            A = vec1
            B= vec2
            distance = np.sqrt(np.sum((A - B) ** 2))

           # print('manhttan distance', scaled_manhattan_distance(vec1, vec2))
            #print("kl distance",kl_div_with_exponential_transform(A,B,moye))

            resul = (1/distance)/x
            #print("Distance euclidienne :", (1/distance)/x)
            bar.progress(100)

            st.markdown(f'La distance euclidienne relatuve est de :red[{round((resul),4)}.]')
        
            if resul > 1 or is_within_10_percent(0.96,resul) == True :
                st.markdown('Il semblerait que votre a été écrit par un humain.')
            elif is_within_10_percent(resul,2) == True :
                st.markdown('Il est sûr que votre texte a été généré.')
            else:
                st.markdown('Il est probable que votre texte a été généré.')

        except:
           st.warning('Un problème est survenu, réessayez ou utilisez un autre module.')


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


    
    with col4:
        
        text_ref = st.text_input("Insérez un/vos textes suspects/ à comparaître dans cette colonne", '')
        pdf_file = st.file_uploader("Télécharger plusieurs textes à comparer au format PDF", type="pdf", accept_multiple_files=True)
     

        if pdf_file is not None:
            for pdf_fil in pdf_file:
            # Lecture du texte de chaque fichier PDF
                pdf_reader = PyPDF2.PdfReader(pdf_fil)
                
                for page in range(len(pdf_reader.pages)):
                        text_ref += pdf_reader.pages[page].extract_text()
        

    if st.button('Vérifier'): #intégrer le texte de référence pour plus de rapidité 
        if text == '' or text_ref=='':
            st.warning("Veuillez indiquer votre texte.")
        else:
                with col1:


                    try :

                        resultatç = lexical_richness_normalized(text, text_ref)

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
                        #print(resul2)

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
                    
                    plt.figure(figsize=(10,5))
                    resul.plot(30, cumulative=False)
                    bar.progress(79) 
                    st.pyplot(plt)
               


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
                    except:
                        try:
                            resul2 = resul.B()
                            resul2 = resul.most_common(resul2)
                            #print(resul2)

                        except:
                                st.warning('Votre texte est trop court.')

                    i = 34

                    df = pd.DataFrame(resul2, columns=['Mots', 'Occurence'])

                    st.dataframe(df)
                    plt.figure(figsize=(10,5))
                    resul.plot(30, cumulative=False)
                    bar.progress(79) 
                    st.pyplot(plt)

                # Liste des coordonnées x et y des points à afficher

                try : 
                    reul = lexical_richness_normalized(text, text_ref)

                    reul = create_markdown_table(reul)
                    st.markdown(reul)

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
                    st.info('Les textes trop courts pour une représentation 2D.')

                bar.progress(100) 
            

elif tabs == "StendhalGPT MultipleTextes":

    st.subheader("StendhalGPT MultipleTextes")
    st.markdown("StendhalGPT MultipleTextes mesure les caractéristiques des textes fournis et les représente dans un plan bidimensionnel.")
    st.info('StendhalGPT MultipleTextes est susceptible d\'évoluer.')


    texte1 = st.text_input("Texte 1")
    texte22 =st.text_input("Texte 2")
    texte3 = st.text_input("Texte 3")
    texte4 = st.text_input("Texte 4")
    texte5 = st.text_input("Texte 5")
    
    resul = [texte1, texte22, texte3, texte4, texte5]

    if st.button("Lancer l'analyse"):
        try:
            calculate_stats(resul)
        except:
            st.warning('Il y a eu une erreur dans le traitement de vos textes.')

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
#        try:
#            anormal_bloc = measure_lexical_richness(text, ponctu)
#            highlighted_text = highlight_text(text, [word for block in anormal_bloc for word in block])
#
#            if anormal_bloc != []:
#                st.markdown("**Blocs Anormaux** : \n" +str(highlighted_text), unsafe_allow_html=True)
#            else : 
#                st.markdown('Votre texte semble être uniforme.')
#        except:
#            st.warning('Un problème est survenu dans le traitement, veuillez choisir une ponctuation différente.')
#        
