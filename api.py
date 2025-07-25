# api.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load model and reference
with open("lung_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_reference.pkl", "rb") as f:
    feature_ref = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data])

        # Ensure all required columns are present
        for col in feature_ref.columns:
            if col not in input_df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(proba, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
