import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Judul & Penjelasan
st.set_page_config(page_title="Deteksi Depresi", page_icon="🧠", layout="centered")
st.title("Aplikasi Deteksi Depresi Mahasiswa")
st.write("Isi 16 parameter di bawah ini secara lengkap untuk memprediksi potensi depresi berdasarkan model Machine Learning yang telah dilatih.")

# 2. Muat Model & Preprocessor dengan Error Handling
try:
    model = joblib.load('model_depresi_final.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('feature_selector.pkl')
except FileNotFoundError as e:
    st.error(f"⚠️ Error: File fisik tidak ditemukan di GitHub. Pastikan file .pkl sudah di-push. Detail: {e}")
    st.stop()

# 3. Form Input (Menyediakan 16 Fitur Sesuai Tuntutan Scaler)
with st.form("main_form"):
    st.subheader("Data Demografi & Akademik")
    col1, col2 = st.columns(2)
    
    with col1:
        # Fitur 1-8
        f1_age = st.number_input("1. Usia", min_value=17, max_value=40, value=20)
        f2_gender = st.selectbox("2. Jenis Kelamin (0=P, 1=L)", [0, 1])
        f3_cgpa = st.number_input("3. Nilai CGPA/IPK", min_value=0.0, max_value=4.0, value=3.5, step=0.1)
        f4_academic_pressure = st.slider("4. Tekanan Akademik (1-5)", 1, 5, 3)
        f5_study_satisfaction = st.slider("5. Kepuasan Belajar (1-5)", 1, 5, 3)
        f6_study_hours = st.number_input("6. Jam Belajar per Hari", min_value=0, max_value=24, value=4)
        f7_financial_stress = st.slider("7. Stres Finansial (1-5)", 1, 5, 2)
        f8_sleep_duration = st.number_input("8. Durasi Tidur per Hari (Jam)", min_value=0, max_value=24, value=7)

    with col2:
        # Fitur 9-16
        f9_suicidal_thoughts = st.selectbox("9. Pikiran Bunuh Diri (0=Tidak, 1=Ya)", [0, 1])
        f10_dietary_habits = st.slider("10. Kualitas Kebiasaan Makan (1-3)", 1, 3, 2)
        f11_family_history = st.selectbox("11. Riwayat Depresi Keluarga (0=Tidak, 1=Ya)", [0, 1])
        f12_work_hours = st.number_input("12. Jam Kerja Part-time/Hari", min_value=0, max_value=24, value=0)
        f13_extracurricular = st.selectbox("13. Aktif Ekstrakurikuler (0=Tidak, 1=Ya)", [0, 1])
        f14_social_support = st.slider("14. Dukungan Sosial (1-5)", 1, 5, 4)
        f15_attendance = st.number_input("15. Persentase Kehadiran (%)", min_value=0, max_value=100, value=90)
        f16_free_time = st.number_input("16. Waktu Luang per Hari (Jam)", min_value=0, max_value=24, value=3)

    submitted = st.form_submit_button("Lakukan Prediksi")

if submitted:
    # 4. Susun input menjadi array (Tepat 16 Elemen)
    data_input = np.array([[
        f1_age, f2_gender, f3_cgpa, f4_academic_pressure, 
        f5_study_satisfaction, f6_study_hours, f7_financial_stress, f8_sleep_duration,
        f9_suicidal_thoughts, f10_dietary_habits, f11_family_history, f12_work_hours,
        f13_extracurricular, f14_social_support, f15_attendance, f16_free_time
    ]]) 
    
    # 5. Keamanan Dimensi
    expected_features = getattr(scaler, 'n_features_in_', None)
    
    if expected_features is not None and data_input.shape[1] != expected_features:
        st.error(f"ERROR DIMENSI: Model dilatih dengan {expected_features} kolom, tetapi menerima {data_input.shape[1]} kolom.")
    else:
        try:
            # 6. Preprocessing (Scaling & Selection)
            data_scaled = scaler.transform(data_input)
            data_final = selector.transform(data_scaled)
            
            # 7. Prediksi
            prediction = model.predict(data_final)
            
            # 8. Tampilkan Hasil
            st.markdown("---")
            if prediction[0] == 1:
                st.error("**Hasil Analisis:** Sistem mendeteksi adanya pola gejala depresi. Disarankan untuk berkonsultasi dengan profesional.")
            else:
                st.success("**Hasil Analisis:** Mahasiswa tidak menunjukkan gejala depresi yang signifikan. Tetap jaga kesehatan mental!")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses perhitungan: {e}")