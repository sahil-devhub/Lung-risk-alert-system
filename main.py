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

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_dict = data.dict()
        print("🔍 Incoming Data:", input_dict)

        df = pd.DataFrame([input_dict])
        df.columns = [col.upper().replace(" ", "_") for col in df.columns]
        print("🧾 DataFrame:", df)

        df = df[feature_ref.columns]  # reordering
        print("✅ Columns matched:", df.columns)

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        return {
            "prediction": prediction,
            "probability": round(probability, 4)
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}
