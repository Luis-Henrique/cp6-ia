# https://6xcxxgtuo4wqfsbqxai4bn.streamlit.app/

import pickle

import numpy as np
import requests
import sklearn
import streamlit as st
from sklearn.preprocessing import StandardScaler

with open('modelo_classificacao_casas.pickle', 'rb') as file:
    model_classificacao = pickle.load(file)

st.title("Aplicação de Classificação e Predição de Casas")

tab1, tab2 = st.tabs(["Previsão de Preço", "Classificação"])

with tab1:
    st.header("Predição de Preço de Casas (via API Flask)")

    sqft_living = st.number_input("Área da casa (sqft_living) para previsão:", min_value=0)
    grade = st.number_input("Grade (avaliação da construção e design):", min_value=1, max_value=13, step=1)
    bathrooms = st.number_input("Número de banheiros (bathrooms) para previsão:", min_value=0.0, step=0.5)
    sqft_above = st.number_input("Área acima do solo (sqft_above):", min_value=0)
    
    if st.button("Prever Preço"):
        payload = {
            "sqft_living": sqft_living,
            "grade": grade,
            "bathrooms": bathrooms,
            "sqft_above": sqft_above
        }
        
        
        response = requests.post("http://0.0.0.0:5000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            predicted_price = result.get("predicted_price", "N/A")
            st.write(f"O preço previsto da casa é: ${predicted_price:,.2f}")
        else:
            st.error("Erro na predição. Verifique os valores inseridos.")

with tab2:
    st.header("Classificação de Casas por Grupos")
    
    sqft_living = st.number_input("Área da casa (sqft_living):", min_value=0)
    bedrooms = st.number_input("Número de quartos (bedrooms):", min_value=0, step=1)
    bathrooms = st.number_input("Número de banheiros (bathrooms):", min_value=0.0, step=0.5)
    floors = st.number_input("Número de andares (floors):", min_value=1, step=1)
    yr_built = st.number_input("Ano de construção (yr_built):", min_value=1800, max_value=2024, step=1)
    
    if st.button("Classificar"):
        features = np.array([[sqft_living, bedrooms, bathrooms, floors, yr_built]])
        
        class_prediction = model_classificacao.predict(features)[0]
        
        st.write(f"A casa pertence ao grupo: {class_prediction}")