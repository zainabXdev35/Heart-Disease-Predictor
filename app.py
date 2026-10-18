"""
Heart Disease Prediction System - Streamlit web app.
Follows Steps 8.1 - 8.5 of the notebook (take input -> feature vector -> encode + scale -> load model -> predict).
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Heart Disease Prediction System", page_icon="❤️")


@st.cache_resource
def load_artifacts():
    """Load the trained model, the scaler and the label encoders saved by the notebook (Step 6.5)."""
    model = pickle.load(open(BASE_DIR / "svc_trained_model.pkl", "rb"))
    scaler = pickle.load(open(BASE_DIR / "scaler.pkl", "rb"))
    encoders = pickle.load(open(BASE_DIR / "label_encoders.pkl", "rb"))
    return model, scaler, encoders


model, scaler, encoders = load_artifacts()

st.title("❤️ Heart Disease Prediction System")
st.write(
    "Enter four measurements from a symptom review and an exercise stress test. "
    "A Support Vector Classifier trained on the UCI Cleveland heart-disease data estimates "
    "whether heart disease is likely."
)

# ---- Step 8.1: take input from the user --------------------------------------------------
chestpain_input = st.selectbox(
    "Chest pain type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
)
maxheartrate_input = st.slider("Maximum heart rate achieved (bpm)", min_value=70, max_value=210, value=150)
exerciseangina_input = st.selectbox("Exercise-induced angina", ["No", "Yes"])
stdepression_input = st.number_input(
    "ST depression induced by exercise (0.0 - 6.5)", min_value=0.0, max_value=6.5, value=1.0, step=0.1
)

if st.button("Predict"):
    # ---- Step 8.2: feature vector (same columns as the training data) ---------------------
    user_input = pd.DataFrame(
        {
            "ChestPain": [chestpain_input],
            "MaxHeartRate": [maxheartrate_input],
            "ExerciseAngina": [exerciseangina_input],
            "STDepression": [stdepression_input],
        }
    )

    # ---- Step 8.3: label-encode, then scale exactly as in training -------------------------
    features = user_input.copy()
    features["ChestPain"] = encoders["ChestPain"].transform(user_input["ChestPain"])
    features["ExerciseAngina"] = encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"])
    features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    # ---- Step 8.5: predict and show the result ---------------------------------------------
    prediction = model.predict(features_scaled)[0]
    if prediction == 1:
        st.error("Prediction: HEART DISEASE LIKELY")
    else:
        st.success("Prediction: NO HEART DISEASE DETECTED")

st.caption(
    "Educational project only - not a medical device and not a diagnosis. "
    "The model uses just 4 inputs and was trained on 303 patients; "
    "it can miss people who do have heart disease. Always consult a doctor."
)
