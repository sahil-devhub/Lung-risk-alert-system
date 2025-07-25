from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open("lung_cancer_model.pkl", "rb"))
feature_ref = pickle.load(open("feature_reference.pkl", "rb"))

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
    return {"message": "Lung Cancer API Running ✅"}

@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.dict()

    # Rename to match training names
    rename_map = {
        "CHRONIC_DISEASE": "CHRONIC DISEASE",
        "ALCOHOL_CONSUMING": "ALCOHOL CONSUMING",
        "SHORTNESS_OF_BREATH": "SHORTNESS OF BREATH",
        "SWALLOWING_DIFFICULTY": "SWALLOWING DIFFICULTY",
        "CHEST_PAIN": "CHEST PAIN"
    }
    for k_old, k_new in rename_map.items():
        input_dict[k_new] = input_dict.pop(k_old)

    df = pd.DataFrame([input_dict])
    df = df[feature_ref.columns]

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {"prediction": prediction, "probability": round(probability, 4)}
