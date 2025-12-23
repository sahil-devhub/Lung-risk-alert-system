# 🫁 Lung Risk Alert System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

An AI-driven health diagnostic tool designed to provide early-stage lung cancer risk assessment. This platform bridges the gap between complex medical screenings and common accessibility, providing instant insights based on lifestyle and symptoms.

---

## 📌 The Problem & Motivation

### The Challenge
Quality healthcare and early diagnostic tests (like CT scans and biopsies) are often **expensive and inaccessible** to many people, especially those from lower-income backgrounds. Because of the high costs and lack of awareness, many individuals ignore early symptoms, leading to late-stage diagnoses where treatment is difficult.

### The Solution
I built the **Lung Risk Alert System** to serve as a "First-Line Screening" tool. 
* **Accessibility:** It’s free and easy to use for anyone with a smartphone/computer.
* **Awareness:** It helps people understand how symptoms like fatigue, wheezing, or even peer pressure (related to smoking) contribute to their risk profile.
* **Actionable Data:** While not a replacement for a doctor, it provides a high-accuracy "Alert" that can encourage a user to seek professional medical help before it's too late.

---

## 📸 Project Gallery

### 🖥️ Landing Page
The user-friendly interface allows anyone to input their data without technical or medical expertise.
![Landing Page](landing.png)

### 📊 Model Performance
The underlying machine learning model is trained for high precision to ensure reliable risk alerts.
![Model Report](model_report.png)

### 🧪 Prediction Output
Instant results with clear indicators of risk levels.
![Prediction Output](predictionoutput%20.png)

### 📈 Data Insights & Grouping
Analyzing how symptoms correlate with lung cancer risk.
![Symptoms Grouping](symptoms%20grouping.png)
![Visual Representation](visual%20representation.png)

---

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** [Streamlit](https://streamlit.io/) (For the interactive Web UI)
* **Machine Learning:** Scikit-Learn, Random Forest Classifier
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Heroku/Streamlit Cloud

---

## 🚀 Key Features

- **Instant Risk Assessment:** Get results in seconds by answering a few lifestyle and health questions.
- **High Accuracy:** Built on a verified survey dataset of lung cancer patients.
- **Categorical Analysis:** Analyzes factors like Age, Gender, Smoking, Anxiety, Peer Pressure, and Chronic Diseases.
- **Zero Cost:** A platform designed for the community to check their health risk without financial burden.

---

## ⚙️ Installation & Usage

To run this project locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sahil-devhub/lung-risk-alert-system.git](https://github.com/sahil-devhub/lung-risk-alert-system.git)
   cd lung-risk-alert-system
   ```
2. **Install dependencies:**
  ```
  pip install -r requirements.txt
  ```
3. **Run the App:**
  ```
  streamlit run app.py
  ```
## 📂 Project Structure
  ```
  ├── app.py                  # Main Streamlit application
  ├── model_training.ipynb    # Jupyter notebook for Model Training & EDA
  ├── lung_cancer_model.pkl   # Pre-trained ML Model
  ├── feature_reference.pkl   # Encoded feature mapping
  ├── requirements.txt        # Project dependencies
  └── assets/                 # Screenshots and images
  ```
## ⚠️ Disclaimer
  ```
  This application is an AI-based risk assessment tool and is intended for educational and awareness purposes only. It does not provide a formal medical diagnosis. Users should always consult with a qualified healthcare professional for medical advice and screenings.
  ```

## 👨‍💻 Developed By
  ```
  Sahil Kumar AI & Machine Learning Enthusiast
  ```


### 💡 Tips for your README:
* **Image Links:** I used the filenames you provided (e.g., `landing.png`). Ensure these images are in the **root folder** of your GitHub repository so they display correctly.
* **Social Links:** Don't forget to update the "Developed By" section with your actual LinkedIn profile link.

**Would you like me to help you write a professional LinkedIn post to share this project now?**
