import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Konfigurasi Halaman (Agar rapi dan tidak terlalu lebar)
st.set_page_config(page_title="Prediksi Depresi", layout="centered")

# 2. Muat Model
try:
    model = joblib.load('model_depresi_final.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('feature_selector.pkl')
except FileNotFoundError:
    st.error("File .pkl tidak ditemukan di direktori GitHub.")
    st.stop()

# --- TAMPILAN UI (Sesuai Screenshot Anda) ---
st.title("Prediksi Depresi Mahasiswa")
st.markdown("---")

st.subheader("Input Data")

# Widget Input (Satu kolom ke bawah)
suicidal_thoughts = st.selectbox("Pernah memiliki pikiran bunuh diri?", ["No", "Yes"])
academic_pressure = st.slider("Academic Pressure", 1, 5, 2)
work_pressure = st.slider("Work Pressure", 1, 5, 2)
financial_stress = st.slider("Financial Stress", 1, 5, 2)
study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
cgpa = st.slider("CGPA", 0.00, 10.00, 5.00, step=0.01)
age = st.slider("Age", 18, 40, 25)
work_study_hours = st.slider("Work/Study Hours", 0, 12, 4)
city = st.selectbox("City", ["Metro", "Tier 2", "Tier 3"])

st.markdown("---")

# Tombol Prediksi
if st.button("Prediksi"):
    
    st.subheader("Hasil")
    
    # Konversi teks ke angka untuk model
    suicidal_val = 1 if suicidal_thoughts == "Yes" else 0
    city_val = 0 if city == "Metro" else (1 if city == "Tier 2" else 2)
    
    # 4. SUSUNAN ARRAY WAJIB (16 Kolom)
    # Ini disesuaikan persis dengan urutan fitur saat df_kotor.drop('Depression') di Notebook Anda
    data_input = np.array([[
        0,                  # 1. Gender (Tumbal)
        age,                # 2. Age
        city_val,           # 3. City
        0,                  # 4. Profession (Tumbal)
        academic_pressure,  # 5. Academic Pressure
        work_pressure,      # 6. Work Pressure
        cgpa,               # 7. CGPA
        study_satisfaction, # 8. Study Satisfaction
        0,                  # 9. Job Satisfaction (Tumbal)
        0,                  # 10. Sleep Duration (Tumbal)
        0,                  # 11. Dietary Habits (Tumbal)
        0,                  # 12. Degree (Tumbal)
        suicidal_val,       # 13. Have you ever had suicidal thoughts ?
        work_study_hours,   # 14. Work/Study Hours
        financial_stress,   # 15. Financial Stress
        0                   # 16. Family History of Mental Illness (Tumbal)
    ]]) 
    
    try:
        # 5. Preprocessing
        data_scaled = scaler.transform(data_input)
        data_final = selector.transform(data_scaled)
        
        # 6. Prediksi Klasifikasi dan Probabilitas
        prediction = model.predict(data_final)
        probabilities = model.predict_proba(data_final)[0]
        
        # Hitung persentase depresi
        depresi_prob = probabilities[1] * 100 
        
        # 7. Tampilkan Hasil
        if prediction[0] == 1:
            st.error("Terindikasi Depresi")
        else:
            st.success("Tidak Depresi")
            
        st.write(f"Probabilitas: **{depresi_prob:.2f}%**")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan pemrosesan: {e}")