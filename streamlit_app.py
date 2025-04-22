import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Student Performance Predictor", page_icon="📙")
st.title("🧑‍🎓 Student Performance Predictor")
st.write("Enter Student details to check the likelihood of Passing.")
study_time = st.slider("📚 Study Time (hrs/day)", 0.0, 10.0, 2.0, 0.5)
attendance = st.slider("📅 Attendance %", 0, 100, 45)
past_grade = st.slider("📝 Past Grade", 0, 100, 45)
parent_support = st.selectbox("👨‍👩‍👧‍👦 Parent Support", ["Yes", "No"])
health = st.slider("🩺 Health (1-poor & 5-Excellent)", 1, 5, 3)
sleep_hrs = st.slider("🛌 Sleep Hours", 0.0, 12.0, 7.0, 0.5)
job = st.selectbox("💼 Part time Job", ["Yes", "No"])
internet = st.selectbox("🌐 Internet Access", ["Yes", "No"])

parent_support_en = 1 if parent_support=="Yes" else 0
job_en = 1 if job=="Yes" else 0
internet_en  = 1 if internet=="Yes" else 0

input_data = np.array([
    study_time, attendance, past_grade, parent_support_en, health, sleep_hrs, job_en, internet_en
]).reshape(1, -1)

scaled_input = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]

    if prediction[0] ==1:
        st.success(f"Likely to Pass _ Confidence: {prob:.2f}")
    else: 
        st.error(f"At Risk of Failing _ Confidence: {prob:.2f}")