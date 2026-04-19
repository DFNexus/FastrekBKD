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

# --- TAMPILAN UI (Sesuai Screenshot) ---
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
    
    # 4. Susunan Array (9 Input dari UI + 7 Angka 0 sebagai "tumbal" agar tidak error dimensi)
    data_input = np.array([[
        suicidal_val, academic_pressure, work_pressure, financial_stress, 
        study_satisfaction, cgpa, age, work_study_hours, city_val, 
        0, 0, 0, 0, 0, 0, 0  # <--- 7 angka nol pengisi ruang
    ]]) 
    
    try:
        # 5. Preprocessing
        data_scaled = scaler.transform(data_input)
        data_final = selector.transform(data_scaled)
        
        # 6. Prediksi Klasifikasi dan Probabilitas
        prediction = model.predict(data_final)
        probabilities = model.predict_proba(data_final)[0] # Mengambil probabilitas
        
        # Hitung persentase depresi (probabilitas kelas 1)
        depresi_prob = probabilities[1] * 100 
        
        # 7. Tampilkan Hasil sesuai screenshot
        if prediction[0] == 1:
            # Jika Depresi (Merah)
            st.error("Terindikasi Depresi")
        else:
            # Jika Tidak Depresi (Hijau)
            st.success("Tidak Depresi")
            
        # Tampilkan teks Probabilitas di bawah kotak warna
        st.write(f"Probabilitas: **{depresi_prob:.2f}%**")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan pemrosesan: {e}")