import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Diabetes Predictor")

inputs = [st.number_input(f) for f in 
['Pregnancies','Glucose','BP','Skin','Insulin','BMI','Pedigree','Age']]

if st.button("Predict"):
    data = scaler.transform([inputs])
    pred = model.predict(data)
    st.success("Diabetic" if pred[0]==1 else "Not Diabetic")
