import streamlit as st
import pandas as pd
import pickle
import os

# Load model
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# Ambil daftar kolom yang digunakan model saat training
model_features = model.feature_names_in_  # Ini hanya ada di scikit-learn >=1.0

st.title("🧠 Employee Attrition Prediction")

# Form input sederhana
age = st.slider("Age", 18, 60, 30)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
overtime = st.selectbox("OverTime", ['Yes', 'No'])

# Bangun input awal
input_dict = {
    'Age': age,
    'MonthlyIncome': income,
    'JobSatisfaction': satisfaction,
    'OverTime_Yes': 1 if overtime == 'Yes' else 0,
    'OverTime_No': 1 if overtime == 'No' else 0,
}

# Ubah ke DataFrame
input_df = pd.DataFrame([input_dict])

# Tambahkan semua fitur kosong yg dibutuhkan model
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Urutkan kolom agar cocok 100% dengan model
input_df = input_df[model_features]

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    st.success(f"Hasil: {'Attrition (Akan Keluar)' if prediction == 1 else 'No Attrition (Tetap)'}")

