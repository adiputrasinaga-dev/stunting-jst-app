import pandas as pd
import os
import glob
import logging

# 1. Konfigurasi Sistem Logging (Mencatat aktivitas ke file etl_pipeline.log)
logging.basicConfig(
    filename='etl_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # 'w' akan menimpa log lama setiap script dijalankan. Gunakan 'a' untuk menyambung.
)

def jalankan_ekstraksi(input_folder, output_file):
    """
    Membaca semua CSV dari input_folder, menggabungkannya, 
    dan menyimpannya ke output_file.
    """
    logging.info(f"MEMULAI PROSES EKSTRAKSI: Mencari file CSV di {input_folder}")
    
    pola_pencarian = os.path.join(input_folder, "*.csv")
    daftar_file = glob.glob(pola_pencarian)
    
    if not daftar_file:
        logging.error("Tidak ada file CSV yang ditemukan! Proses dihentikan.")
        return None
        
    logging.info(f"Ditemukan {len(daftar_file)} file CSV. Memulai penggabungan...")
    
    list_dataframe = []
    
    for file in daftar_file:
        nama_file = os.path.basename(file)
        logging.info(f"-> Membaca file: {nama_file}")
        
        try:
            df_sementara = pd.read_csv(file)
            list_dataframe.append(df_sementara)
        except Exception as e:
            logging.error(f"Gagal membaca {nama_file}. Error: {e}")
            
    # Menggabungkan semua data
    df_gabungan = pd.concat(list_dataframe, ignore_index=True)
    logging.info(f"Penggabungan selesai. Total baris data: {len(df_gabungan)}")
    logging.info(f"Kolom yang tersedia: {', '.join(df_gabungan.columns)}")
    
    # Memastikan folder tujuan ada (jika belum, Python akan membuatnya otomatis)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Menyimpan data hasil ekstrak ke folder processed agar bisa dibaca script ke-2
    df_gabungan.to_csv(output_file, index=False)
    logging.info(f"PROSES EKSTRAKSI SELESAI: Data berhasil disimpan ke {output_file}")
    
    return df_gabungan

if __name__ == "__main__":
    # Menentukan jalur folder input dan output
    folder_sumber = os.path.join("data", "raw")
    file_tujuan = os.path.join("data", "processed", "01_extracted_data.csv")
    
    # Menjalankan fungsi
    df_hasil = jalankan_ekstraksi(folder_sumber, file_tujuan)
    
    # Tampilkan pesan ke layar agar kita tahu script sudah selesai berjalan
    print("✅ Proses ekstraksi selesai! Silakan buka file 'etl_pipeline.log' untuk melihat detailnya.")