import requests
import streamlit as st
import pandas as pd
import numpy as np

st.title('StendhalGPT')
token = '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'
API_URL = "https://api-inference.huggingface.co/models/roberta-large-openai-detector"
headers = {"Authorization": f"Bearer {token}"}
st.text('Cet outil est en version de développement.' )
def query(payload):
    
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

default_value_goes_here = 'Quel est votre texte ? (Anglais Uniquement).'
user_input = st.text_input("Votre texte ici.", default_value_goes_here)



if st.button('Vérifier'):
    output = query({
	"inputs": user_input,
        })
    try:
        if output[0][1]['score'] >=  output[0][0]['score']:
            st.text('Votre texte est à'+ str(round(output[0][0]['score'], 2)*100)+'% de chance d\'être écrit par un homme.')    
        else : 
            st.text('Votre texte est à '+ str(round(output[0][1]['score'], 2)*100)+'% de chance d\'être génèré par une IA.' ) 
    except:
        st.text('Le service est temporairement indisponible, merci de votre compréhension.' )


