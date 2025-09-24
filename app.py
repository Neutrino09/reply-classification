from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("baseline_model.joblib")

# FastAPI app
app = FastAPI(title="Reply Classification API")

# Request schema
class InputText(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(input: InputText):
    text = [input.text]
    proba = model.predict_proba(text)[0]
    label = model.predict(text)[0]
    confidence = float(np.max(proba))
    return {"label": label, "confidence": round(confidence, 3)}
