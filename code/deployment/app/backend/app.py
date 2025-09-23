# app.py

from fastapi import FastAPI

from pydantic import BaseModel

import pickle

# Load the trained model

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app

app = FastAPI()


# Define the input data schema

class IrisInput(BaseModel):
    sepal_length: float

    sepal_width: float

    petal_length: float

    petal_width: float


# Define the prediction endpoint

@app.post("/predict")
def predict(input_data: IrisInput):
    data = [[

        input_data.sepal_length,

        input_data.sepal_width,

        input_data.petal_length,

        input_data.petal_width

    ]]

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}

