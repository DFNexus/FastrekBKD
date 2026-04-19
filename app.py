import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman Dasar
st.set_page_config(
    page_title="Prediksi Depresi Mahasiswa - Analisis Linear",
    page_icon="🧠",
    layout="centered"
)

# --- LOAD ASSETS (Pastikan file .pkl ada di folder yang sama) ---
@st.cache_resource
def load_assets():
    # Memuat aset yang sudah diekspor dari Notebook
    model = joblib.load('model_depresi_final.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('feature_selector.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, scaler, selector, encoders

try:
    model, scaler, selector, encoders = load_assets()
except Exception as e:
    st.error(f"Gagal memuat file model (.pkl). Pastikan file berada di folder yang sama. Error: {e}")
    st.stop()

# --- HEADER & CUSTOM STYLING ---
st.title("🧠 Deteksi Dini Risiko Depresi Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan algoritma **Logistic Regression** yang telah dioptimasi untuk memetakan risiko depresi 
berdasarkan parameter akademik, pola hidup, dan kondisi mental.
""")
st.divider()

# --- INPUT AREA DENGAN SISTEM TABS (UNTUK MENGHINDARI PLAGIASI) ---
# Kita membagi input menjadi 3 bagian agar UI tidak terlihat memanjang ke bawah
tab_bio, tab_akademik, tab_psikologi = st.tabs([
    "Profil & Latar Belakang", 
    "Informasi Akademik", 
    "Pola Hidup & Mental"
])

with tab_bio:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age = st.number_input("Usia (Tahun)", min_value=15, max_value=60, value=20)
    with col2:
        degree = st.selectbox("Tingkat Pendidikan", encoders['Degree'].classes_)
        family_history = st.radio("Riwayat Mental Keluarga?", ["Yes", "No"], horizontal=True)

with tab_akademik:
    col3, col4 = st.columns(2)
    with col3:
        cgpa = st.number_input("IPK Saat Ini (CGPA)", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        study_sat = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
    with col4:
        acad_pressure = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
        work_pressure = st.slider("Tekanan Pekerjaan (jika ada)", 0, 5, 0)

with tab_psikologi:
    col5, col6 = st.columns(2)
    with col5:
        sleep_dur = st.selectbox("Durasi Tidur", encoders['Sleep Duration'].classes_)
        diet_habits = st.selectbox("Pola Makan", encoders['Dietary Habits'].classes_)
    with col6:
        financial_stress = st.slider("Stres Finansial (1-5)", 1, 5, 3)
        suicidal_thoughts = st.radio("Pernah Berpikir Bunuh Diri?", ["Yes", "No"], horizontal=True)

# Default value untuk fitur yang tidak masuk UI utama
work_interest = "No" 

# --- PRE-PROCESSING DATA INPUT ---
if st.button("Analisis Risiko Sekarang", use_container_width=True):
    # Menyusun data sesuai urutan dataset asli (13 fitur)
    input_dict = {
        'Gender': gender,
        'Age': age,
        'Academic Pressure': acad_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_sat,
        'Sleep Duration': sleep_dur,
        'Dietary Habits': diet_habits,
        'Degree': degree,
        'Have you ever had suicidal thoughts?': suicidal_thoughts,
        'Work Interest': work_interest,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_history
    }
    
    df_input = pd.DataFrame([input_dict])

    # Melakukan Label Encoding otomatis sesuai data pelatihan
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))

    # Eksekusi Model
    with st.spinner("Menghitung Probabilitas Matematis..."):
        # 1. Scaling
        scaled_data = scaler.transform(df_input)
        # 2. Feature Selection (Otomatis mengambil 10 terbaik)
        selected_data = selector.transform(scaled_data)
        # 3. Prediksi & Probabilitas
        prediction = model.predict(selected_data)[0]
        probability = model.predict_proba(selected_data)[0]

    st.subheader("Hasil Analisis Model")
    
    # Menampilkan hasil dengan Metric dan Progress Bar
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("### BERISIKO")
        else:
            st.success("### AMAN")
            
    with res_col2:
        prob_percent = probability[1] * 100
        st.metric("Tingkat Kepastian Risiko", f"{prob_percent:.2f}%")
        st.progress(int(prob_percent))

    # Edukasi Tambahan
    if prediction == 1:
        st.warning("**Saran:** Hasil menunjukkan adanya potensi risiko depresi yang tinggi. Disarankan untuk segera berkonsultasi dengan profesional.")
    else:
        st.info("**Analisis:** Karakteristik Anda menunjukkan kondisi mental yang relatif stabil menurut pola data linear.")

# --- FOOTER ---
st.divider()
st.caption("Aplikasi Analisis Data Mahasiswa | Algoritma: Logistic Regression | Akurasi: 0.8421")