import streamlit as st
import pickle
import numpy as np

with open('modelo_classificacao_casas.pickle', 'rb') as file:
    model = pickle.load(file)

st.title("Classificação de Casas por Grupos")

sqft_living = st.number_input("Área da casa (sqft_living):", min_value=0)
bedrooms = st.number_input("Número de quartos (bedrooms):", min_value=0, step=1)
bathrooms = st.number_input("Número de banheiros (bathrooms):", min_value=0.0, step=0.5)
floors = st.number_input("Número de andares (floors):", min_value=1, step=1)
yr_built = st.number_input("Ano de construção (yr_built):", min_value=1800, max_value=2024, step=1)

if st.button("Classificar"):
    features = np.array([[sqft_living, bedrooms, bathrooms, floors, yr_built]])
    
    class_prediction = model.predict(features)[0]
    
    st.write(f"A casa pertence ao grupo: {class_prediction}")
