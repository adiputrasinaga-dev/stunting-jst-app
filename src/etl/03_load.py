import pandas as pd
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# 1. Konfigurasi Sistem Logging
logging.basicConfig(
    filename='etl_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # 'a' agar log terakumulasi dari proses extract dan transform
)

def jalankan_load(input_file, db_connection_string, nama_tabel):
    logging.info("MEMULAI PROSES LOAD DATA KE CLOUD DATABASE...")
    
    try:
        # 1. Membaca data hasil transformasi (02_transformed_data.csv)
        if not os.path.exists(input_file):
            logging.error(f"File sumber tidak ditemukan: {input_file}")
            return False
            
        df = pd.read_csv(input_file)
        logging.info(f"Data siap-load berhasil dimuat: {len(df)} baris.")
        
        # 2. Membuat koneksi (Engine) ke PostgreSQL Neon.tech
        engine = create_engine(db_connection_string)
        
        logging.info(f"Menghubungkan ke Neon.tech dan memasukkan data ke tabel '{nama_tabel}'...")
        
        # 3. Mengirimkan data dari Pandas Dataframe langsung ke PostgreSQL
        # if_exists='append' memastikan data masuk ke struktur tabel yang sudah kita buat di SQL Editor
        df.to_sql(name=nama_tabel, con=engine, if_exists='append', index=False)
        
        logging.info("PROSES LOAD SELESAI: Data sukses bermigrasi ke Cloud Database!\n")
        return True

    except SQLAlchemyError as e:
        logging.error(f"Error Koneksi/Query Database: {e}")
        return False
    except Exception as e:
        logging.error(f"Terjadi kesalahan sistem pada tahap Load: {e}")
        return False

if __name__ == "__main__":
    # Jalur file data bersih hasil tahap 02_transform
    file_sumber = os.path.join("data", "processed", "02_transformed_data.csv")
    
    # ⚠️ PASTE KEMBALI CONNECTION STRING DARI NEON.TECH KAMU DI SINI
    # Contoh format: "postgresql://username:password@ep-namaserver.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
    DB_URL = "postgresql://neondb_owner:npg_KlTnJ0GFDIh1@ep-bold-fire-ao1htius-pooler.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    
    # Nama tabel yang disesuaikan dengan skema database
    NAMA_TABEL = "riwayat_pengukuran_balita"
    
    print("Mengirimkan data ke Cloud Database Neon.tech... (Pantau 'etl_pipeline.log' untuk detail)")
    
    if DB_URL == "MASUKKAN_CONNECTION_STRING_NEON_MU_DI_SINI":
        print("❌ Gagal: Kamu belum memasukkan Connection String dari Neon.tech pada variabel DB_URL.")
    else:
        sukses = jalankan_load(file_sumber, DB_URL, NAMA_TABEL)
        if sukses:
            print("✅ Yey! Seluruh data berhasil masuk ke Cloud Database Neon.tech!")
            print("Silakan cek menu 'Tables' di dashboard Neon.tech untuk melihat datamu.")
        else:
            print("❌ Proses load gagal. Silakan periksa file 'etl_pipeline.log' untuk melihat detail error.")