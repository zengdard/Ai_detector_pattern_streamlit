import streamlit as st
st.set_page_config(layout="wide")

st.title('StendhalGPT')
from st_on_hover_tabs import on_hover_tabs

from stendhalgpt_fct import *
import pandas as pd


import nltk




col3, col4 = st.columns(2)
col1, col2 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8, col9 = st.columns(3)

with st.sidebar:
        tabs = on_hover_tabs(tabName=['Accueil','StendhalGPT', 'StendhalGPT Expert', 'StendhalGPT MultipleTextes'], 
                            iconName=['dashboard','home',  'toll', 'analytics'],
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



def compare_markov_model(text1, text2):
    # tokenize les deux textes
    tokens1 = nltk.word_tokenize(text1)
    tokens2 = nltk.word_tokenize(text2)

    # créer des bigrames pour les deux textes
    bigrams1 = list(bigrams(tokens1))
    bigrams2 = list(bigrams(tokens2))

    # compter le nombre d'occurences de chaque bigramme  A MODIFIER PAR LA TF IDF
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



with open('TEXTE.txt', "r", encoding="utf-8") as f:
     content = f.read()




if tabs == 'Accueil':
    st.info('Si vous rencontrez des difficultés, vous pouvez nous contacter dans la page Contact de notre site.')

    st.markdown(content)


    st.markdown('[Cliquez ici](https://www.stendhalgpt.fr/newsletter/) pour vous inscrire à la newsletter.')
    st.markdown('[Cliquez ici](https://www.stendhalgpt.fr/docs-category/doc/) pour accéder à la documentation.')

    st.caption('version 0.6.0')

elif tabs == 'StendhalGPT':
    bar.progress(0) 

    st.info('En dessous de 130 mots, il est préférable d\'utiliser la fonction Expert. ')

    
    with col5:

        text = st.text_input("Insérez un/vos texte(s) référent dans cette colonne.", '')
    
    
    with col6:
        nbr_mots_text = len(text.split(" "))
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
            diff = compare_markov_model(nettoyer_texte(text), nettoyer_texte(text_ref))
            vec1 = np.array([diff[0][bigram] for bigram in diff[0]] +[verbal_richness(text)]+[grammatical_richness(text)]+[lexical_richness(text)] )
            vec2 = np.array([diff[1][bigram] for bigram in diff[1]] +[verbal_richness(text_ref)]+[grammatical_richness(text_ref)]+[lexical_richness(text_ref)])
                        
            x = len(vec1)
            A = vec1
            B= vec2
            distance = np.sqrt(np.sum((A - B) ** 2))
            resul = (1/distance)/x

        # print('manhttan distance', scaled_manhattan_distance(vec1, vec2))
            #print("kl distance",kl_div_with_exponential_transform(A,B,moye))

            #print("Distance euclidienne :", (1/distance)/x)
            bar.progress(100)

            st.markdown(f'La distance euclidienne relative est de :red[{round((resul),4)}.]')
        
            if resul > 1 or is_within_10_percent(0.96,resul) == True :
                st.markdown('Il semblerait que votre a été écrit par un humain.')
            elif is_within_10_percent(resul,2) == True :
                st.markdown('Il est sûr que votre texte a été généré.')
            else:
                st.markdown('Il est probable que votre texte a été généré.')

        except:
            st.warning('Un problème est survenu, réessayez ou utilisez un autre module.')


elif tabs == 'StendhalGPT Expert':
    st.info('Vous utilisez actuellement StendhalGPT Expert')
    with col3:


        text = st.text_input("Insérez un/vos texte(s) référent dans cette colonne.", '')
        pdf_file2 = st.file_uploader("Télécharger plusieurs textes de référence au format PDF", type="pdf", accept_multiple_files=False)

        if pdf_file2 is not None:
            for pdf_fil2 in pdf_file2:
                # Lecture du texte de chaque fichier PDF
                pdf_reader = PyPDF2.PdfReader(pdf_fil2)
                for page in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page].extract_text()
    with col4:
        
        text_ref = st.text_input("Insérez un/vos textes suspects/ à comparaître dans cette colonne", '')
        pdf_file = st.file_uploader("Télécharger plusieurs textes à comparer au format PDF", type="pdf", accept_multiple_files=False)
    

        if pdf_file is not None:
            for pdf_fil in pdf_file:
            # Lecture du texte de chaque fichier PDF
                pdf_reader = PyPDF2.PdfReader(pdf_fil)
                
                for page in range(len(pdf_reader.pages)):
                        text_ref += pdf_reader.pages[page].extract_text()
        

    if st.button('Vérifier'): #intégrer le texte de référence pour plus de rapidité 
                with col1:
                    bar.progress(0) 
                    

                    try :


                        richesse_gram = round(grammatical_richness(text),4)

                        richesse_lex = round(lexical_richness(text),4)
                        richesse_verbale = round(verbal_richness(text),4)


                        compteur_mots = count_words(text)
                        texte_metrique = round(compute_text_metrics(richesse_lex,richesse_gram,richesse_verbale ),4)

                        nrb = count_characters(text)
 
                        st.markdown("### Résultat Texte 1 :")
                        st.markdown(f"Taux correspondant au taux lexical de votre texte(s) : **{richesse_lex}** ")
                        st.markdown(f"Taux correspondant au taux grammatical de votre texte(s) : **{richesse_gram}** ")
                        st.markdown(f"Taux correspondant au taux verbal de votre texte(s) : **{richesse_verbale}** ") 
                        st.markdown(f"Nombre de mots : **{compteur_mots}** ")  

                        st.markdown(f"Nombre de caractères: **{nrb}** ")  

                        st.markdown(f"### Indicateur de richesse générale :  :red[{round(texte_metrique * 100,4)}]")  


                    except:
                        st.warning("Votre texte est trop court.")

                    #Fréquence des mots 
                    resul = lexical_field(text)
                    try:
                        resul2 = resul.B()
                        resul2 = resul.most_common(resul2)

                    except:
                        try:
                            resul2 = resul.B()
                            print(resul2)
                            resul2 = resul.most_common(resul2)

                        except:
                            st.warning('Votre texte est trop court.')
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

                        st.markdown("### Résultat Texte 2:")
                        st.markdown(f"Taux correspondant au taux lexical de votre texte(s) : **{richesse_lex2}** ")
                        st.markdown(f"Taux correspondant au taux grammatical de votre texte(s) : **{richesse_gram2}** ")
                        st.markdown(f"Taux correspondant au taux verbal de votre texte(s) : **{richesse_verbale2}** ") 
                        st.markdown(f"Nombre de mots : **{compteur_mots}** ")   

                        st.markdown(f"Nombre de caractères: **{nrb}** ")  

                        st.markdown(f"### Indicateur de richesse :  :red[{round(texte_metrique2 * 100,4)}]")  
                        





                    except:
                        st.warning("Votre texte est trop court.")
                    

                    resul = lexical_field(text_ref)
                    
                    try:
                        resul2 = resul.B()
                        resul2 = resul.most_common(resul2)
                    except:
                        
                        st.warning('Votre texte est trop court.')

                    i = 34

                    df = pd.DataFrame(resul2, columns=['Mots', 'Occurence'])

                    st.dataframe(df)
                    plt.figure(figsize=(10,5))
                    resul.plot(30, cumulative=False)
                    bar.progress(79) 
                    st.pyplot(plt)





                    with col7:
                        try : 
                            text_dt= lexical_richness_normalized_ttr(text)
                            ref_text_dt = lexical_richness_normalized_ttr(text_ref)

                            #[unique1,ttr1,mtd1]

                            plot_text_relations_2d([text_dt[0], text_dt[1]], [ref_text_dt[0], ref_text_dt[1]])

                            P1 =  np.array([text_dt[0], text_dt[1]])
                            P2 = np.array([ref_text_dt[0], ref_text_dt[1]])

                            dist=  np.sqrt(np.sum((P1 - P2) ** 2))
                            st.markdown(f"Distance entre les points {dist}.")
                        except:
                            st.warning("Une erreur est survenue dans le traitement de vos textes.")
                    with col8:
                        try: 


                            text_dt_tt = lexical_richness_normalized(text)
                            text_ref_dt_tt = lexical_richness_normalized(text_ref)



                            P1 =  np.array(text_dt_tt)
                            P2 = np.array(text_ref_dt_tt)

                            dist=  np.sqrt(np.sum((P1 - P2) ** 2))

                            text1 = text_dt_tt
                            text2 = text_ref_dt_tt

                            texts = [text1, text2]

                            plot_text_relations(texts)

                            st.markdown(f"Distance Euclidienne entre les points {dist}.")
                        except:
                                st.warning("Une erreur est survenue dans le traitement de vos textes.")


                            
                    with col9:
                        try: 
                            plot_texts_3d((richesse_lex,richesse_gram,richesse_verbale),(richesse_lex2,richesse_gram2,richesse_verbale2))

                            P1 =  np.array([richesse_lex, richesse_gram,richesse_verbale])
                            P2 = np.array([richesse_lex2, richesse_gram2,richesse_verbale2])

                            dist=  np.sqrt(np.sum((P1 - P2) ** 2))
                            st.markdown(f"Distance Euclidienne entre les points {dist}.")

                        except:
                           st.warning('Une erreur est survenue dans le traitement de vos textes.')

                     
                try:
                          
                    diff = compare_markov_model(nettoyer_texte(text), nettoyer_texte(text_ref))
                    vec1 = np.array([diff[0][bigram] for bigram in diff[0]] +[verbal_richness(text)]+[grammatical_richness(text)]+[lexical_richness(text)] )
                    vec2 = np.array([diff[1][bigram] for bigram in diff[1]] +[verbal_richness(text_ref)]+[grammatical_richness(text_ref)]+[lexical_richness(text_ref)])
                                
                    x = len(vec1)
                    distance_eucli = np.sqrt(np.sum((vec1 - vec2) ** 2))

                    if x == 0 or distance_eucli == 0 :
                        resultat_euli = 0
                    else:
                        resultat_euli = (1/distance_eucli)/x
                    st.header(f'La distance euclidienne normalisée entre les deux textes est  :red[{resultat_euli}.]')

                except:
                    st.warning('Textes trop courts pour une mesure de distance euclidienne.')
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

            plot_text_relations(lexical_richness_normalized(texte1),lexical_richness_normalized(texte22),lexical_richness_normalized(texte3),lexical_richness_normalized(texte4),lexical_richness_normalized(texte5))

            plot_texts_3d(lexical_richness_normalized_only(texte1),lexical_richness_normalized_only(texte22),lexical_richness_normalized_only(texte3),lexical_richness_normalized_only(texte4),lexical_richness_normalized_only(texte5))


        except:
          st.warning('Il y a eu une erreur dans le traitement de vos textes.')
 