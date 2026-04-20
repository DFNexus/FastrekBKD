import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman Dasar
st.set_page_config(
    page_title="Deteksi Depresi Mahasiswa",
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
st.title("Deteksi Depresi Mahasiswa")
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
        # slider
        sleep_dur = st.slider("Durasi Tidur (Jam)", 1, 12, 7) 
    with col6:
        financial_stress = st.slider("Stres Finansial (1-5)", 1, 5, 3)
        suicidal_thoughts = st.radio("Pernah Berpikir Bunuh Diri?", ["Yes", "No"], horizontal=True)

# --- LOGIKA PREDIKSI ---
if st.button("Jalankan Analisis", use_container_width=True):

    if sleep_dur < 5:
        sleep_str = "Less than 5 hours"
    elif sleep_dur <= 6:
        sleep_str = "5-6 hours"
    elif sleep_dur <= 8:
        sleep_str = "7-8 hours"
    else:
        sleep_str = "More than 8 hours"
    
    # Default value fitur tambahan
    input_dict = {
        'Gender': gender,
        'Age': age,
        'Academic Pressure': acad_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_sat,
        'Sleep Duration': sleep_str, # Menggunakan string hasil mapping
        'Dietary Habits': "Moderate",
        'Degree': "BSc",
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work Interest': 0,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': "No",
        'City': encoders['City'].classes_[0] if 'City' in encoders else 0,
        'Profession': encoders['Profession'].classes_[0] if 'Profession' in encoders else 0,
        'Job Satisfaction': 3,
        'Work/Study Hours': 5
    }

    df_input = pd.DataFrame([input_dict])

    # --- ENCODING ---
    for col, le in encoders.items():
        if col in df_input.columns:
            val = str(df_input[col].iloc[0])

            # fallback kalau value tidak dikenal
            if val not in le.classes_:
                val = le.classes_[0]

            df_input[col] = le.transform([val])[0]

    # --- CONVERT NUMERIK ---
    data_numeric = df_input.apply(pd.to_numeric, errors='coerce')
    if data_numeric.isnull().any().any():
        # Mengekstrak nama kolom pasti yang bernilai NaN
        kolom_error = data_numeric.columns[data_numeric.isnull().any()].tolist()
        st.error(f"FATAL: Terdapat string yang gagal dikonversi menjadi angka. Kolom penyebab: {kolom_error}")
        st.warning(f"Fakta: Kolom {kolom_error} berisi teks, namun namanya tidak ditemukan di dalam 'label_encoders.pkl' Anda. Pastikan nama kolom di input sama persis dengan yang ada di model training.")
        st.write("Data mentah bermasalah:", df_input[kolom_error])
        st.stop()

    # --- SAMAKAN URUTAN FITUR ---
    try:
        data_numeric = data_numeric[scaler.feature_names_in_]
    except Exception as e:
        st.error(f"Urutan fitur tidak sesuai: {e}")
        st.stop()

    # --- PREDIKSI ---
    with st.spinner("Menganalisis data..."):
        scaled_data = scaler.transform(data_numeric)
        selected_data = selector.transform(scaled_data)

        prediction = model.predict(selected_data)[0]
        probability = model.predict_proba(selected_data)[0]

    # --- HASIL ---
    st.subheader("Hasil Analisis")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("### BUDREK (Depresi)")
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