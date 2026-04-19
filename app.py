import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Judul & Penjelasan (Syarat Poin 58)
st.title("Aplikasi Deteksi Depresi Mahasiswa")
st.write("Gunakan formulir ini untuk memprediksi potensi tingkat depresi berdasarkan data akademik.")

# 2. Muat Model & Preprocessor (Syarat Poin 56)
model = joblib.load('model_depresi_final.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('feature_selector.pkl')

# 3. Form Input (Syarat Poin 57)
with st.form("main_form"):
    st.subheader("Input Data Mahasiswa")
    # Contoh input (Ganti dengan 10 fitur terbaik hasil SelectKBest Anda)
    age = st.number_input("Usia", min_value=18, max_value=35, value=20)
    academic_pressure = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
    study_satisfaction = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
    
    submitted = st.form_submit_button("Lakukan Prediksi")

if submitted:
    # 4. Susun input menjadi array (Pastikan urutan kolom SAMA dengan saat training)
    # Anda harus memasukkan semua 10 fitur di sini
    data_input = np.array([[age, academic_pressure, study_satisfaction, 0, 0, 0, 0, 0, 0, 0]]) 
    
    # 5. Preprocessing (Scaling & Selection)
    data_scaled = scaler.transform(data_input)
    data_final = selector.transform(data_scaled)
    
    # 6. Prediksi
    prediction = model.predict(data_final)
    
    # 7. Tampilkan Hasil (Syarat Poin 57)
    if prediction[0] == 1:
        st.error("Hasil: Mahasiswa Terdeteksi Mengalami Gejala Depresi.")
    else:
        st.success("Hasil: Mahasiswa Tidak Terdeteksi Mengalami Gejala Depresi.")