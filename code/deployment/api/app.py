from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline



classifier = pipeline("text-classification", model="models/classifier")
generator = pipeline("text2text-generation", model="models/generator")

app = FastAPI(title="Tech Assistant API")

class InputQuestion(BaseModel):
    text: str

@app.post("/classify")
def classify_question(data: InputQuestion):
    results = classifier(data.text)
    # возвращаем топ-3 категорий
    return {"classification": results}


@app.post("/generate")
def generate_answer(data: InputQuestion):
    answer = generator(data.text, max_length=50)
    return {"answer": answer[0]['generated_text']}
