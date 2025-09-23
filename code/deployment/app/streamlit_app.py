# streamlit_app.py

import streamlit as st

import requests

# FastAPI endpoint

FASTAPI_URL = "http://fastapi:8000/predict"

# Streamlit app UI

st.title("Iris Flower Classifier")

# Input fields for the Iris flower data

sepal_length = st.number_input("Sepal Length", min_value=0.0)

sepal_width = st.number_input("Sepal Width", min_value=0.0)

petal_length = st.number_input("Petal Length", min_value=0.0)

petal_width = st.number_input("Petal Width", min_value=0.0)

# Make prediction when the button is clicked

if st.button("Predict"):
    # Prepare the data for the API request

    input_data = {

        "sepal_length": sepal_length,

        "sepal_width": sepal_width,

        "petal_length": petal_length,

        "petal_width": petal_width

    }

    # Send a request to the FastAPI prediction endpoint

    response = requests.post(FASTAPI_URL, json=input_data)

    prediction = response.json()["prediction"]

    # Display the result

    st.success(f"The model predicts class: {prediction}")

