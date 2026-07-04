import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Prediksi Stunting JST",
    page_icon="👶",
    layout="centered"
)

# 2. Fungsi untuk memuat model dan scaler (di-cache agar web tidak lemot saat di-refresh)
@st.cache_resource
def load_ml_components():
    # Pastikan path ini sesuai dengan lokasi komputermu
    model_path = os.path.join('models', 'export', 'champion_model_jst.keras')
    scaler_path = os.path.join('models', 'export', 'minmax_scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("❌ Model atau Scaler tidak ditemukan! Pastikan proses training (Fase 2) sudah selesai.")
        return None, None
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_ml_components()

# 3. Desain Antarmuka Pengguna (UI)
st.title("🩺 Deteksi Dini Stunting Balita")
st.markdown("""
Aplikasi cerdas berbasis **Jaringan Saraf Tiruan (Backpropagation)** untuk memprediksi status stunting pada balita berdasarkan pengukuran fisik. 
*(Model dioptimasi dengan Adam Optimizer - Tingkat Akurasi 99.8%)*
""")
st.divider()

# Membuat form input di web
col1, col2 = st.columns(2)

with col1:
    jk_input = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur_input = st.number_input("Umur (Bulan)", min_value=0, max_value=60, value=24)

with col2:
    tinggi_input = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=120.0, value=85.0, format="%.1f")

# 4. Logika Prediksi saat tombol ditekan
if st.button("🔍 Analisis Status Balita", use_container_width=True):
    if model is not None and scaler is not None:
        with st.spinner("Sedang memproses data dengan Jaringan Saraf Tiruan..."):
            # Konversi input teks ke angka sesuai Feature Engineering kita
            jk_val = 1 if jk_input == "Laki-laki" else 0
            
            # Susun array input
            X_new = np.array([[jk_val, umur_input, tinggi_input]])
            
            # Wajib di-scale menggunakan scaler dari data training! (Anti Leakage)
            X_new_scaled = scaler.transform(X_new)
            
            # Prediksi menggunakan model (karena sigmoid, outputnya probabilitas 0-1)
            probabilitas = model.predict(X_new_scaled)[0][0]
            
            st.divider()
            
            # Logika Output Visual
            if probabilitas > 0.5:
                st.error(f"⚠️ **HASIL: INDICATED STUNTING** (Tingkat Keyakinan: {probabilitas*100:.1f}%)")
                st.warning("Rekomendasi: Segera konsultasikan asupan gizi balita dengan dokter anak atau petugas puskesmas terdekat.")
            else:
                st.success(f"✅ **HASIL: NORMAL** (Tingkat Keyakinan: {(1-probabilitas)*100:.1f}%)")
                st.info("Rekomendasi: Pertahankan pola asuh dan asupan gizi seimbang.")