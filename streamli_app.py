import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model_rf.pkl', 'rb'))

st.title("Employee Attrition Prediction")

# Input fitur pengguna
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
over_time = st.selectbox("OverTime", ['Yes', 'No'])

# Preprocess input
over_time_yes = 1 if over_time == 'Yes' else 0

input_df = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'JobSatisfaction': [job_satisfaction],
    'OverTime_Yes': [over_time_yes]
})

# Prediksi
prediction = model.predict(input_df)[0]
st.write("Prediction:", "Attrition" if prediction == 1 else "No Attrition")
