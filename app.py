import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman Dasar
st.set_page_config(
    page_title="Prediksi Depresi Mahasiswa",
    page_icon="",
    layout="centered"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
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

# --- HEADER ---
st.title("Deteksi Risiko Depresi Mahasiswa")
st.markdown("""
Aplikasi ini menggunakan algoritma **Logistic Regression** untuk memetakan risiko depresi 
berdasarkan parameter akademik dan kondisi psikologis.
""")
st.divider()

# --- INPUT AREA (TABS MODIFIKASI) ---
tab_profil, tab_akademik, tab_psikologi = st.tabs([
    "Profil Pengguna", 
    "Informasi Akademik", 
    "Kondisi Mental"
])

with tab_profil:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    with col2:
        age = st.number_input("Usia (Tahun)", min_value=15, max_value=60, value=20)

with tab_akademik:
    col3, col4 = st.columns(2)
    with col3:
        cgpa = st.number_input("IPK Saat Ini (CGPA)", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        study_sat = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
    with col4:
        acad_pressure = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
        work_pressure = st.slider("Tekanan Pekerjaan (0-5)", 0, 5, 0)

with tab_psikologi:
    col5, col6 = st.columns(2)
    with col5:
        sleep_dur = st.selectbox("Durasi Tidur", encoders['Sleep Duration'].classes_)
    with col6:
        financial_stress = st.slider("Stres Finansial (1-5)", 1, 5, 3)
        suicidal_thoughts = st.radio("Pernah Berpikir Bunuh Diri?", ["Yes", "No"], horizontal=True)

# --- LOGIKA PREDIKSI ---
if st.button("Jalankan Analisis", use_container_width=True):
    
    # Nilai Default untuk fitur yang dihapus dari UI (Wajib ada agar dimensi data cocok)
    diet_habits = "Moderate"
    degree = "BSc"
    work_interest = "No"
    family_history = "No"

    # Menyusun data sesuai urutan dataset asli (13 fitur)
    # Perhatikan nama kolom "Have you ever had suicidal thoughts ?" (menggunakan spasi sebelum ?)
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
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work Interest': work_interest,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_history
    }
    
    df_input = pd.DataFrame([input_dict])

    # Melakukan Label Encoding otomatis
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))

    st.write(df_input.dtypes)
    st.write(df_input)

    # Eksekusi Model
    with st.spinner("Menganalisis data..."):
        # Pastikan data berupa numerik murni
        data_numeric = df_input.apply(pd.to_numeric, errors='coerce')
        
        # 1. Scaling
        scaled_data = scaler.transform(data_numeric)
        # 2. Feature Selection
        selected_data = selector.transform(scaled_data)
        # 3. Prediksi
        prediction = model.predict(selected_data)[0]
        probability = model.predict_proba(selected_data)[0]

    # --- TAMPILAN HASIL ---
    st.subheader("Hasil Analisis")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("### BERISIKO")
        else:
            st.success("### AMAN")
            
    with res_col2:
        prob_percent = probability[1] * 100
        st.metric("Tingkat Kepastian", f"{prob_percent:.2f}%")
        st.progress(int(prob_percent))

    if prediction == 1:
        st.warning("**Saran:** Segera konsultasikan kondisi Anda dengan ahli kesehatan mental.")
    else:
        st.info("**Hasil:** Kondisi Anda tergolong stabil menurut analisis linear.")

st.divider()
st.caption("Analisis Data Mahasiswa | Logistic Regression | Akurasi: 0.8421")