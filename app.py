import streamlit as st
import pandas as pd
import joblib

# Load the trained model
artifact = joblib.load("placement_model.pkl")

model = artifact["model"]
features = artifact["features"]
model_name = artifact["model_name"]

# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
h1 {
    text-align: center;
    color: #222222;
}
div.stButton > button {
    width: 100%;
    background-color: #087cf0;
    color: white;
    border-radius: 6px;
    height: 45px;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# Application heading
st.title("Student Placement Predictor")
st.write(
    "Predict student placement using academic performance "
    "and skills."
)

st.info("This model uses pre-interview student features.")

# Academic information
st.subheader("Academic Information")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "CGPA (out of 10)",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

with col2:
    internship = st.selectbox(
        "Internship completed?",
        ["Yes", "No"]
    )

# Projects and certifications
st.subheader("Projects and Certifications")

col1, col2 = st.columns(2)

with col1:
    projects = st.number_input(
        "Number of projects",
        min_value=0,
        max_value=100,
        value=3
    )

with col2:
    certifications = st.number_input(
        "Number of certifications",
        min_value=0,
        max_value=100,
        value=2
    )

# Skills
st.subheader("Skills and Aptitude")

col1, col2 = st.columns(2)

with col1:
    communication = st.slider(
        "Communication Skills",
        0.0, 10.0, 7.0
    )
    programming = st.slider(
        "Programming Skills",
        0.0, 10.0, 7.0
    )

with col2:
    technical = st.slider(
        "Technical Skills",
        0.0, 10.0, 7.0
    )
    aptitude = st.slider(
        "Aptitude Score",
        0.0, 100.0, 70.0
    )

# Prediction
if st.button("Predict Placement"):

    student = pd.DataFrame([{
        "cgpa": cgpa,
        "internship": 1 if internship == "Yes" else 0,
        "communication_skills": communication,
        "certifications": certifications,
        "projects": projects,
        "programming_skills": programming,
        "technical_skills": technical,
        "aptitude": aptitude
    }])[features]

    prediction = int(model.predict(student)[0])
    probability = float(
        model.predict_proba(student)[0][1]
    )

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Outcome: PLACED")
    else:
        st.warning("Predicted Outcome: NOT PLACED")

    st.metric(
        "Estimated Placement Probability",
        f"{probability * 100:.1f}%"
    )

    st.progress(probability)

    st.caption(f"Model used: {model_name}")
    st.caption(
        "This is an educational estimate, not a guarantee "
        "of real-world placement."
    )

import sklearn
import sys

st.write("Python version:", sys.version)
st.write("Scikit-learn version:", sklearn.__version__)
