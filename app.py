import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Header
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient health details below to predict diabetes risk using a KNN classifier trained on the Pima Indians Diabetes Dataset.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=130, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (μU/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=18, max_value=90, value=30)

st.divider()

# Predict button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ High Risk — This patient is likely **Diabetic**")
        st.metric("Diabetic Probability", f"{probability[1]*100:.1f}%")
    else:
        st.success("✅ Low Risk — This patient is likely **Not Diabetic**")
        st.metric("Non-Diabetic Probability", f"{probability[0]*100:.1f}%")

    # Show confidence bar
    st.markdown("#### Prediction Confidence")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Not Diabetic", f"{probability[0]*100:.1f}%")
    with col_b:
        st.metric("Diabetic", f"{probability[1]*100:.1f}%")

    st.progress(float(probability[1]))

st.divider()
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for medical advice.")
st.caption("Built by [Sridivya](https://github.com/Diviya-tech) | Model: KNN (K=5) | Dataset: Pima Indians Diabetes")
