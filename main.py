from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

app = FastAPI()

# Define the origins that are allowed to make requests to your API.
# It's crucial to include the domain where your Hopweb app will be hosted.
# For testing, you can use "*" to allow all origins, but for production,
# specify the exact domains for better security.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://your-hopweb-app-domain.com", # <--- IMPORTANT: Replace with your actual Hopweb app domain
    "https://lung-risk-alert-system.onrender.com", # If your frontend is also hosted here
    # If running the HTML file directly from your local machine (file://),
    # the origin might be 'null'. You can add it for local testing,
    # but be cautious with it in production as it's less secure.
    # "null"
    "*" # For development/testing, allows all origins. REMOVE OR RESTRICT FOR PRODUCTION!
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of allowed origins
    allow_credentials=True,      # Allow cookies/authentication headers to be sent
    allow_methods=["*"],         # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],         # Allow all headers
)

# Load your model and feature reference
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
    try:
        input_dict = data.dict()

        # Rename to match training
        rename_map = {
            "CHRONIC_DISEASE": "CHRONIC DISEASE",
            "ALCOHOL_CONSUMING": "ALCOHOL CONSUMING",
            "SHORTNESS_OF_BREATH": "SHORTNESS OF BREATH",
            "SWALLOWING_DIFFICULTY": "SWALLOWING DIFFICULTY",
            "CHEST_PAIN": "CHEST PAIN"
        }
        for old_key, new_key in rename_map.items():
            input_dict[new_key] = input_dict.pop(old_key)

        df = pd.DataFrame([input_dict])
        df = df[feature_ref.columns]

        # Predict
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        # Add risk levels
        if probability >= 0.7:
            risk_level = "high"
            recommendation = "⚠️ Urgent: Please consult a doctor immediately."
            status_color = "red"
        elif probability >= 0.3:
            risk_level = "moderate"
            recommendation = "⚠️ Caution: Monitor symptoms and consider screening."
            status_color = "orange"
        else:
            risk_level = "low"
            recommendation = "✅ You are likely safe. Maintain regular check-ups."
            status_color = "green"

        return {
            "prediction": "High Risk" if prediction == 1 else "Low Risk",
            "probability": f"{round(probability * 100, 2)}%",
            "risk_level": risk_level,
            "recommendation": recommendation,
            "status_color": status_color
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

