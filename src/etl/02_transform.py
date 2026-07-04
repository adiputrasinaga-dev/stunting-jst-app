import pandas as pd
import numpy as np
import logging
import os
import uuid
from datetime import datetime

# 1. Konfigurasi Sistem Logging
logging.basicConfig(
    filename='etl_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a' 
)

# 2. Fungsi Anotasi (Diadaptasi dari skripsimu)
def tentukan_status_stunting(tb_u):
    stunting_category = {"Sangat Pendek", "Pendek"}
    non_stunting_category = {"Normal", "Tinggi"}
    
    if pd.isna(tb_u):
        return "-"
    if tb_u in stunting_category:
        return "Stunting"
    if tb_u in non_stunting_category:
        return "Normal" 
    return "-"

# 3. Fungsi IQR Outlier (Diadaptasi dari skripsimu)
def bersihkan_outliers_iqr(df, kolom, k=1.5):
    Q1 = df[kolom].quantile(0.25)
    Q3 = df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - k * IQR
    batas_atas = Q3 + k * IQR
    return df[(df[kolom] >= batas_bawah) & (df[kolom] <= batas_atas)]

def jalankan_transformasi(input_file, output_file):
    logging.info("MEMULAI PROSES TRANSFORMASI DATA...")
    
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Data awal dimuat: {len(df)} baris.")
        
        # A. DATA ANNOTATION
        if 'TB/U' in df.columns:
            df['Status'] = df['TB/U'].apply(tentukan_status_stunting)
        else:
            logging.warning("Kolom 'TB/U' tidak ditemukan!")
            
        # B. DATA CLEANING
        df['Tinggi Badan'] = pd.to_numeric(df['Tinggi Badan'], errors='coerce')
        df = df.dropna(subset=['Tinggi Badan'])
        df = df[df['Status'] != '-']
        df = df.drop_duplicates()
        
        # C. OUTLIER REMOVAL 
        df_normal = df[df['Status'] == 'Normal']
        df_stunting = df[df['Status'] == 'Stunting']
        
        # Ini baris yang tadi error karena ketambahan tag referensi
        df_bersih_normal = bersihkan_outliers_iqr(df_normal, 'Tinggi Badan', k=1.4)
        df_bersih_stunting = bersihkan_outliers_iqr(df_stunting, 'Tinggi Badan', k=1.4)
        df = pd.concat([df_bersih_normal, df_bersih_stunting])
        
        logging.info(f"Setelah Data Cleaning & Outlier Removal: {len(df)} baris tersisa.")
        
        # D. PENYESUAIAN SKEMA DATABASE 
        df['id_balita'] = [f"BALITA-{str(uuid.uuid4())[:8].upper()}" for _ in range(len(df))]
        df['tanggal_ukur'] = datetime.today().strftime('%Y-%m-%d')
        df['jenis_kelamin'] = df['Jenis Kelamin'].map({'L': 'Laki-laki', 'P': 'Perempuan'})
        
        df_final = pd.DataFrame({
            'id_balita': df['id_balita'],
            'tanggal_ukur': df['tanggal_ukur'],
            'jenis_kelamin': df['jenis_kelamin'],
            'umur_bulan': df['Umur'].astype(int),
            'berat_badan_kg': df.get('Berat Badan', pd.Series([np.nan]*len(df))), 
            'tinggi_badan_cm': df['Tinggi Badan'].round(2),
            'status_stunting': df['Status']
        })
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_final.to_csv(output_file, index=False)
        logging.info(f"PROSES TRANSFORMASI SELESAI: Data siap-database disimpan ke {output_file}\n")
        
        return df_final

    except Exception as e:
        logging.error(f"Terjadi kesalahan saat transformasi: {e}")
        return None

if __name__ == "__main__":
    file_sumber = os.path.join("data", "processed", "01_extracted_data.csv")
    file_tujuan = os.path.join("data", "processed", "02_transformed_data.csv")
    
    df_hasil = jalankan_transformasi(file_sumber, file_tujuan)
    
    if df_hasil is not None:
        print("✅ Proses Transformasi Selesai!")
        print("\n--- 5 Baris Pertama Data Siap Database ---")
        print(df_hasil.head())
        print(f"\nTotal data siap dimuat ke Database: {len(df_hasil)} baris.")