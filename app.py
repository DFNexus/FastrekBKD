import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Judul & Penjelasan
st.set_page_config(page_title="Deteksi Depresi", page_icon="🧠")
st.title("Aplikasi Deteksi Depresi Mahasiswa")
st.write("Gunakan formulir ini untuk memprediksi potensi tingkat depresi berdasarkan data akademik dan gaya hidup.")

# 2. Muat Model & Preprocessor dengan Error Handling
try:
    model = joblib.load('model_depresi_final.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('feature_selector.pkl')
except FileNotFoundError as e:
    st.error(f"Error: File fisik tidak ditemukan di GitHub. Pastikan file .pkl sudah di-push. Detail: {e}")
    st.stop()

# 3. Form Input
with st.form("main_form"):
    st.subheader("Input Data Mahasiswa")
    
    # --- BAGIAN INPUT PENGGUNA ---
    # WAJIB DIUBAH: Tambahkan widget input di sini sesuai dengan seluruh kolom X_train Anda
    age = st.number_input("Usia", min_value=18, max_value=35, value=20)
    academic_pressure = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
    study_satisfaction = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
    # Contoh input tambahan (Hapus tanda pagar jika ini fitur Anda):
    # cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.0)
    # sleep_duration = st.selectbox("Durasi Tidur", ["Kurang", "Cukup", "Berlebih"])
    
    submitted = st.form_submit_button("Lakukan Prediksi")

if submitted:
    # 4. Susun input menjadi array
    # WAJIB DIUBAH: Masukkan semua nama variabel input dari form di atas ke dalam array ini.
    # Urutannya HARUS PERSIS SAMA dengan urutan kolom saat Anda melatih model di Google Colab.
    data_input = np.array([[
        age, 
        academic_pressure, 
        study_satisfaction,
        0, 0, 0, 0, 0, 0, 0 # Ganti angka 0 ini dengan variabel input Anda yang lain
    ]]) 
    
    # 5. Keamanan Dimensi (Mencegah ValueError Crash)
    expected_features = getattr(scaler, 'n_features_in_', None)
    
    if expected_features is not None and data_input.shape[1] != expected_features:
        st.error(f"ERROR DIMENSI: Model Anda dilatih dengan {expected_features} kolom, tetapi Anda memasukkan {data_input.shape[1]} kolom.")
        st.warning(f"Solusi: Tambahkan {expected_features - data_input.shape[1]} variabel input lagi di kode baris ke-43 (`data_input`).")
    else:
        try:
            # 6. Preprocessing (Scaling & Selection)
            data_scaled = scaler.transform(data_input)
            data_final = selector.transform(data_scaled)
            
            # 7. Prediksi
            prediction = model.predict(data_final)
            
            # 8. Tampilkan Hasil
            if prediction[0] == 1:
                st.error("Hasil: Mahasiswa Terdeteksi Mengalami Gejala Depresi.")
            else:
                st.success("Hasil: Mahasiswa Tidak Terdeteksi Mengalami Gejala Depresi.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses perhitungan matematis: {e}")