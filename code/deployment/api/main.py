import logging
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api ")

classifier = pipeline("text-classification", model="models/classifier", top_k=3)
generator = pipeline("text2text-generation", model="models/generator")

app = FastAPI(title="Tech Assistant API")

class InputQuestion(BaseModel):
    text: str

@app.post("/classify")
def classify_question(data: InputQuestion):
    result = classifier(data.text)
    logger.info("Classifier results type: %s", type(result))
    logger.info("Classifier results: %s", result)

    classification = result[0] if isinstance(result, list) else result
    if isinstance(classification, list):
        result = max(classification, key=lambda x: x["score"])
    else:
        result = classification

    # return the result
    return {"label": result["label"], "score": result["score"]}


@app.post("/generate")
def generate_answer(data: InputQuestion):
    answer = generator(data.text, max_length=250)
    logger.info("Generator results type: %s", type(answer))
    logger.info("Generator results: %s", answer)
    return {"answer": answer[0]['generated_text']}
