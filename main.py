from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load model and feature reference
with open("lung_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_reference.pkl", "rb") as f:
    feature_ref = pickle.load(f)

app = FastAPI()

class PatientData(BaseModel):
    AGE: int
    GENDER: int
    SMOKING: int
    YELLOW_FINGERS: int
    ANXIETY: int
    PEER_PRESSURE: int
    CHRONIC_DISEASE: int
    FATIGUE: int
    ALLERGY: int
    WHEEZING: int
    ALCOHOL_CONSUMING: int
    COUGHING: int
    SHORTNESS_OF_BREATH: int
    SWALLOWING_DIFFICULTY: int
    CHEST_PAIN: int

@app.get("/")
def home():
    return {"message": "Lung Cancer Predictor API is Live 🔥"}

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])
    # Make sure the column names match
    df.columns = [col.upper().replace(" ", "_") for col in df.columns]
    df = df[feature_ref.columns]  # match training order
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }
