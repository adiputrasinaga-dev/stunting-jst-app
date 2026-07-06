import os
import numpy as np
import pandas as pd
import joblib
import logging
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================
# 🏛️ 1. DATA REFERENSI PERMENKES NO. 2 TAHUN 2020 (-2 SD)
# Diekstrak langsung dari Lampiran Standar Antropometri Anak
# ============================================================
# Batas minimal tinggi badan normal (cm) per bulan (0-60 bulan)
BATAS_STUNTING_LAKI = [
    46.1, 50.8, 54.4, 57.3, 59.7, 61.7, 63.3, 64.8, 66.2, 67.5, 68.7, 69.9, 
    71.0, 72.1, 73.1, 74.1, 75.0, 76.0, 76.9, 77.7, 78.6, 79.4, 80.2, 81.0, 
    81.0, 81.7, 82.5, 83.1, 83.8, 84.5, 85.1, 85.7, 86.4, 86.9, 87.5, 88.1, 
    88.7, 89.2, 89.8, 90.3, 90.9, 91.4, 91.9, 92.4, 92.9, 93.4, 93.9, 94.4, 
    94.9, 95.4, 95.9, 96.4, 96.9, 97.4, 97.8, 98.3, 98.8, 99.3, 99.7, 100.2, 100.7
]

BATAS_STUNTING_PEREMPUAN = [
    45.4, 49.8, 53.0, 55.6, 57.8, 59.6, 61.2, 62.7, 64.0, 65.3, 66.5, 67.7, 
    68.9, 70.0, 71.0, 72.0, 73.0, 74.0, 74.9, 75.8, 76.7, 77.5, 78.4, 79.2, 
    79.3, 80.0, 80.8, 81.5, 82.2, 82.9, 83.6, 84.3, 84.9, 85.6, 86.2, 86.8, 
    87.4, 88.0, 88.6, 89.2, 89.8, 90.4, 90.9, 91.5, 92.0, 92.5, 93.1, 93.6, 
    94.1, 94.6, 95.1, 95.6, 96.1, 96.6, 97.1, 97.6, 98.1, 98.5, 99.0, 99.5, 99.9
]

def validasi_klinis_vs_jst(db_url):
    print("⚖️ MEMULAI PENGUJIAN VALIDASI KLINIS: PERMENKES 2/2020 VS JST AI")
    
    # 2. Tarik Seluruh Data dari Cloud Database
    engine = create_engine(db_url)
    query = "SELECT jenis_kelamin, umur_bulan, tinggi_badan_cm FROM riwayat_pengukuran_balita"
    df = pd.read_sql(query, con=engine)
    print(f"✅ Menarik {len(df)} data balita dari database untuk diuji...")

    # 3. Hitung Y_TRUE (Kebenaran Mutlak berdasarkan Hukum Permenkes)
    y_true_permenkes = []
    for _, row in df.iterrows():
        jk = row['jenis_kelamin']
        umur = int(row['umur_bulan'])
        tinggi = row['tinggi_badan_cm']
        
        # Keamanan jika ada data umur di atas 60 bulan, kita asumsikan batas 60 bulan
        umur = 60 if umur > 60 else (0 if umur < 0 else umur)
        
        batas_kritis = BATAS_STUNTING_LAKI[umur] if jk == 'Laki-laki' else BATAS_STUNTING_PEREMPUAN[umur]
        
        # 1 = Stunting (Tinggi di bawah batas), 0 = Normal
        if tinggi < batas_kritis:
            y_true_permenkes.append(1)
        else:
            y_true_permenkes.append(0)
            
    df['aktual_permenkes'] = y_true_permenkes

    # 4. Hitung Y_PRED (Prediksi dari Model JST)
    model_path = os.path.join('models', 'export', 'champion_model_jst.keras')
    scaler_path = os.path.join('models', 'export', 'minmax_scaler.joblib')
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Feature Engineering untuk model
    df['jk_angka'] = df['jenis_kelamin'].map({'Perempuan': 0, 'Laki-laki': 1})
    X_uji = df[['jk_angka', 'umur_bulan', 'tinggi_badan_cm']].values
    X_uji_scaled = scaler.transform(X_uji)
    
    print("🧠 Sedang meminta AI menebak status ribuan balita...")
    y_prob = model.predict(X_uji_scaled, verbose=0).flatten()
    y_pred_jst = (y_prob > 0.5).astype(int)
    
    # 5. KALKULASI METRIK EVALUASI MEDIS
    acc = accuracy_score(y_true_permenkes, y_pred_jst)
    prec = precision_score(y_true_permenkes, y_pred_jst, zero_division=0)
    rec = recall_score(y_true_permenkes, y_pred_jst, zero_division=0)
    f1 = f1_score(y_true_permenkes, y_pred_jst, zero_division=0)

    print("\n" + "="*50)
    print("🏆 HASIL VALIDASI KLINIS (PERMENKES VS JST) 🏆")
    print("="*50)
    print(f"Kepatuhan Model thd Aturan Negara (Accuracy) : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Tingkat Alarm Palsu (Precision)            : {prec:.4f}")
    print(f"Kemampuan Mendeteksi Pasien (Recall)       : {rec:.4f}")
    print(f"Harmoni Prediksi Klinis (F1-Score)         : {f1:.4f}")
    print("="*50)

    # 6. Plot Confusion Matrix Medis
    cm = confusion_matrix(y_true_permenkes, y_pred_jst)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Normal (AI)', 'Stunting (AI)'], 
                yticklabels=['Normal (Hukum)', 'Stunting (Hukum)'],
                annot_kws={'size': 14})
    plt.title('Validasi Klinis: Hukum Kemenkes vs AI Prediksi', fontsize=14, fontweight='bold')
    
    fig_path = os.path.join('reports', 'figures', 'clinical_validation_matrix.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"🎨 Visualisasi Matriks Pengadilan Klinis disimpan di: {fig_path}")

if __name__ == "__main__":
    # ⚠️ Masukkan connection string dari Neon.tech milikmu di sini
    KONEKSI_DB = "postgresql://neondb_owner:npg_KlTnJ0GFDIh1@ep-bold-fire-ao1htius-pooler.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    
    if KONEKSI_DB == "MASUKKAN_CONNECTION_STRING_NEON_MU_DI_SINI":
        print("❌ Gagal: Masukkan Connection String Neon.tech kamu dulu.")
    else:
        validasi_klinis_vs_jst(KONEKSI_DB)