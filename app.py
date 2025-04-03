import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained model
model_file = "lung_cancer_model.pkl"
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Load feature reference to ensure correct column names
with open("feature_reference.pkl", "rb") as file:
    feature_ref = pickle.load(file)

# Streamlit app
st.title("Lung Risk Alert System")
st.write("This app predicts the likelihood of lung cancer based on patient information.")

# Sidebar for feature importance
st.sidebar.title("Model Information")
importances = model.feature_importances_
feature_names = feature_ref.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
st.sidebar.write("Feature Importances:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
plt.tight_layout()
st.sidebar.pyplot(fig)

# Main app layout
col1, col2 = st.columns(2)
with col1:
    st.header("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    st.header("Smoking and Habits")
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["No", "Yes"])
with col2:
    st.header("Symptoms")
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    st.header("Other Factors")
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])

# Convert inputs to model format
input_data = pd.DataFrame({
    "GENDER": [1 if gender == "Male" else 0],
    "AGE": [age],
    "SMOKING": [1 if smoking == "Yes" else 0],
    "YELLOW_FINGERS": [1 if yellow_fingers == "Yes" else 0],
    "ANXIETY": [1 if anxiety == "Yes" else 0],
    "PEER_PRESSURE": [1 if peer_pressure == "Yes" else 0],
    "CHRONIC DISEASE": [1 if chronic_disease == "Yes" else 0],
    "FATIGUE": [1 if fatigue == "Yes" else 0],
    "ALLERGY": [1 if allergy == "Yes" else 0],
    "WHEEZING": [1 if wheezing == "Yes" else 0],
    "ALCOHOL CONSUMING": [1 if alcohol_consumption == "Yes" else 0],
    "COUGHING": [1 if coughing == "Yes" else 0],
    "SHORTNESS OF BREATH": [1 if shortness_of_breath == "Yes" else 0],
    "SWALLOWING DIFFICULTY": [1 if swallowing_difficulty == "Yes" else 0],
    "CHEST PAIN": [1 if chest_pain == "Yes" else 0]
})

# Ensure we have exactly the same columns as during training
for col in feature_ref.columns:
    if col not in input_data.columns:
        st.error(f"Missing column: {col}")

# Define symptom categories and their weights for probability calibration
# Group symptoms into primary, secondary, and tertiary categories
primary_symptoms = {
    "SMOKING": 3.5,        # Major risk factor
    "CHRONIC DISEASE": 3.0,
}

secondary_symptoms = {
    "CHEST PAIN": 2.0,
    "SHORTNESS OF BREATH": 2.0,
    "COUGHING": 1.5,       # Reduced from 2.0 to 1.5
    "WHEEZING": 1.5,
    "YELLOW_FINGERS": 1.5,
    "SWALLOWING DIFFICULTY": 1.5,
}

tertiary_symptoms = {
    "FATIGUE": 0.8,
    "ALCOHOL CONSUMING": 0.7,
    "ANXIETY": 0.4,
    "ALLERGY": 0.3,
    "PEER_PRESSURE": 0.2
}

# Combine all symptoms for convenience
all_symptoms = {**primary_symptoms, **secondary_symptoms, **tertiary_symptoms}

# Count symptoms by category
def count_symptom_categories(input_data_df):
    primary_count = sum(1 for symptom in primary_symptoms if input_data_df[symptom].iloc[0] == 1)
    secondary_count = sum(1 for symptom in secondary_symptoms if input_data_df[symptom].iloc[0] == 1)
    tertiary_count = sum(1 for symptom in tertiary_symptoms if input_data_df[symptom].iloc[0] == 1)
    return primary_count, secondary_count, tertiary_count

# Calculate weighted symptom score with emphasis on combinations
def calculate_advanced_symptom_score(input_data_df, weights_dict):
    # Basic weighted score
    symptom_score = 0
    max_possible_score = sum(weights_dict.values())
    
    # Count how many symptoms are present by category
    primary_count, secondary_count, tertiary_count = count_symptom_categories(input_data_df)
    total_symptom_count = primary_count + secondary_count + tertiary_count
    
    # Calculate basic weighted score
    for symptom, weight in weights_dict.items():
        if symptom in input_data_df.columns and input_data_df[symptom].iloc[0] == 1:
            symptom_score += weight
    
    # Normalize score (0-1)
    normalized_score = symptom_score / max_possible_score if max_possible_score > 0 else 0
    
    # Apply adjustments for single symptoms or limited combinations
    if total_symptom_count == 0:
        return 0.0  # No symptoms
    elif total_symptom_count == 1:
        # Single symptom case - cap at low value depending on which category
        if primary_count == 1:
            return min(normalized_score * 0.3, 0.1)  # Single primary symptom
        elif secondary_count == 1:
            return min(normalized_score * 0.2, 0.05)  # Single secondary symptom (like coughing)
        else:
            return min(normalized_score * 0.1, 0.02)  # Single tertiary symptom
    elif primary_count == 0 and secondary_count <= 2:
        # 1-2 secondary symptoms without primary risk factors
        return min(normalized_score * 0.4, 0.15)
    elif primary_count == 0:
        # Multiple secondary/tertiary symptoms but no primary risk factors
        return min(normalized_score * 0.6, 0.25)
    else:
        # At least one primary symptom plus others - more realistic risk
        # Scale based on total count and mix of symptoms
        if total_symptom_count >= 5:
            # Multiple symptoms including primary - potentially high risk
            return normalized_score * 0.9
        else:
            # More limited combination
            return normalized_score * 0.7
            
# Calculate age risk factor (increases with age)
def calculate_age_factor(age):
    if age < 30:
        return 0.4  # Lower risk for young people
    elif age < 45:
        return 0.6  # Below average risk
    elif age < 60:
        return 0.8  # Moderate risk
    elif age < 70:
        return 1.0  # Standard risk
    else:
        return 1.2  # Higher risk for elderly

# Apply model calibration with clinically appropriate scaling
def calibrate_probability(raw_probability, symptom_score, age_factor, gender_factor=1.0):
    # First, adjust for extremely low or no symptoms
    if symptom_score < 0.01:  # Essentially no symptoms
        return 0.01  # Minimal baseline risk
    
    # Apply progressive scaling based on symptom score ranges
    if symptom_score < 0.05:  # Very minimal symptoms
        base_prob = min(0.03, raw_probability * 0.1)
    elif symptom_score < 0.1:  # Very few symptoms 
        base_prob = min(0.05, raw_probability * 0.15)
    elif symptom_score < 0.2:  # Few minor symptoms
        base_prob = min(0.1, raw_probability * 0.25)
    elif symptom_score < 0.3:  # Some concerning symptoms
        base_prob = min(0.2, raw_probability * 0.4)
    elif symptom_score < 0.4:  # Moderate symptom level
        base_prob = min(0.35, raw_probability * 0.6)
    elif symptom_score < 0.6:  # More significant symptoms
        base_prob = min(0.6, raw_probability * 0.8)
    else:  # High level of concerning symptoms
        base_prob = min(0.9, raw_probability * 0.95)
    
    # Apply age and gender adjustment factors
    adjusted_prob = base_prob * age_factor * gender_factor
    
    # Ensure probability stays in valid range
    return min(max(adjusted_prob, 0.01), 0.95)  # Cap between 1% and 95%

# Gender risk adjustment (males have higher risk)
def calculate_gender_factor(gender):
    return 1.2 if gender == "Male" else 0.9

# Predict button
if st.button("Predict"):
    try:
        # Get probability from the model
        probability = model.predict_proba(input_data)
        raw_cancer_probability = probability[0][1]  # Positive class probability
        
        # Calculate advanced symptom score
        symptom_score = calculate_advanced_symptom_score(input_data, all_symptoms)
        
        # Calculate adjustment factors
        age_factor = calculate_age_factor(age)
        gender_factor = calculate_gender_factor(gender)
        
        # Apply calibration
        cancer_probability = calibrate_probability(
            raw_cancer_probability, 
            symptom_score, 
            age_factor, 
            gender_factor
        )
        
        # Get symptom counts for debugging
        primary_count, secondary_count, tertiary_count = count_symptom_categories(input_data)
        total_count = primary_count + secondary_count + tertiary_count
        
        # Display debug information in sidebar
        st.sidebar.subheader("Debug Information")
        st.sidebar.write(f"Raw model probability: {raw_cancer_probability:.4f}")
        st.sidebar.write(f"Symptom score: {symptom_score:.4f}")
        st.sidebar.write(f"Primary symptoms: {primary_count}/{len(primary_symptoms)}")
        st.sidebar.write(f"Secondary symptoms: {secondary_count}/{len(secondary_symptoms)}")
        st.sidebar.write(f"Tertiary symptoms: {tertiary_count}/{len(tertiary_symptoms)}")
        st.sidebar.write(f"Age factor: {age_factor:.2f}")
        st.sidebar.write(f"Gender factor: {gender_factor:.2f}")
        st.sidebar.write(f"Calibrated probability: {cancer_probability:.4f}")

        # Display results based on calibrated probability
        if cancer_probability >= 0.7:
            st.error(f"High Risk: The patient is likely to have lung cancer. (Probability: {cancer_probability:.2%})")
        elif cancer_probability >= 0.3:
            st.warning(f"Moderate Risk: The patient has some risk factors for lung cancer. (Probability: {cancer_probability:.2%})")
        else:
            st.success(f"Low Risk: The patient is unlikely to have lung cancer. (Probability: {cancer_probability:.2%})")

        # Recommendations section
        st.subheader("Recommendations:")
        recommendations = []
        
        # Base recommendations on specific symptoms and their combinations
        if smoking == "Yes":
            recommendations.append("- Consider smoking cessation programs")
            
        if coughing == "Yes" and (shortness_of_breath == "Yes" or wheezing == "Yes"):
            recommendations.append("- Monitor respiratory symptoms and maintain a symptom diary")
            
        if primary_count >= 1 and secondary_count >= 1:
            recommendations.append("- Consult with a healthcare provider for evaluation")
            
        if (primary_count >= 1 and secondary_count >= 2) or cancer_probability >= 0.3:
            recommendations.append("- Consider screening tests such as low-dose CT scan")
            
        if cancer_probability >= 0.7 or (smoking == "Yes" and chest_pain == "Yes" and shortness_of_breath == "Yes"):
            recommendations.append("- Urgent: Seek immediate medical consultation")
            recommendations.append("- Diagnostic testing should be prioritized")
            
        if total_count == 0 or cancer_probability < 0.1:
            recommendations.append("- Maintain regular health check-ups")
            
        # Add age-specific recommendations
        if age >= 55 and smoking == "Yes":
            recommendations.append("- Annual lung cancer screening recommended for your age and smoking history")
        
        # Display all recommendations
        for rec in recommendations:
            st.write(rec)

        # Gauge chart
        st.subheader("Cancer Risk Assessment")
        gauge_fig, gauge_ax = plt.subplots(figsize=(5, 6), subplot_kw=dict(polar=True))
        cmap = plt.cm.RdYlGn_r
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.ones_like(theta)
        norm = plt.Normalize(0, 1)
        colors = cmap(np.linspace(0, 1, 100))
        gauge_ax.bar(theta, r, width=2 * np.pi / 100, color=colors, alpha=0.5)
        
        # Needle
        needle_theta = 2 * np.pi * cancer_probability
        gauge_ax.plot([0, needle_theta], [0, 0.9], 'k-', lw=3)
        gauge_ax.plot([needle_theta], [0.9], 'ko', ms=10)
        
        # Remove ticks and spines
        gauge_ax.set_xticks([])
        gauge_ax.set_yticks([])
        gauge_ax.spines['polar'].set_visible(False)
        
        # Labels
        plt.text(0, 1.2, "Low Risk", ha='center', va='center', fontsize=10)
        plt.text(np.pi/2, 1.2, "Low/Moderate", ha='center', va='center', fontsize=10)
        plt.text(np.pi, 1.2, "Moderate/High", ha='center', va='center', fontsize=10)
        plt.text(3*np.pi/2, 1.2, "High Risk", ha='center', va='center', fontsize=10)
        plt.text(0, 0, f"{cancer_probability:.2%}", ha='center', va='center', fontsize=20)
        
        plt.title("Cancer Risk Assessment", fontsize=14)
        st.pyplot(gauge_fig)

    except Exception as e:
        st.error(f"Error: {e}")