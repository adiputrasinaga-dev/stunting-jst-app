import os
import random
import time
import logging
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine

# Library Machine Learning & Pencocokan Parameter Jurnal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

# ============================================================
# 🔧 1. REPRODUCIBILITY SETUP (Persis Aturan R2 di Jurnal)
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

# Logging Setup
logging.basicConfig(
    filename='ml_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Weights awal dari eksperimen Jurnal untuk replikasi sempurna
MODEL_INITIAL_WEIGHTS = [
    np.array([
        [0.2988949, 0.15892175, -0.23386581, -0.3509909, -0.16581696, 0.3143032, -0.15123937, -0.07626566, -0.26894704, -0.3985059, 0.4132057, -0.14021748, 0.29796162, 0.20572975, 0.0548279, 0.29503563, -0.3388698, 0.3681161, 0.41373393, -0.10563782, 0.15669742, 0.21421042, -0.36973026, -0.08551681, -0.02134064, 0.16800418, 0.3332027, 0.24919483, -0.14808118, -0.291131, 0.2186512, -0.39212063],
        [0.3415391, -0.13676414, 0.0340001, -0.1587401, 0.3891417, 0.1140947, 0.01216254, 0.2133061, 0.17810991, 0.39973018, -0.06172618, -0.4080195, -0.38629124, -0.27222675, 0.20937148, 0.04213923, 0.20766923, -0.3581156, 0.03880247, 0.37105432, -0.38384667, 0.11059806, -0.03844661, 0.3841928, -0.2881276, 0.13132992, -0.00523967, -0.1505143, -0.10458186, -0.28216907, -0.33551094, -0.03285611],
        [0.3575857, -0.34065092, -0.29343903, 0.03563541, 0.31690767, 0.2984015, 0.21953431, -0.09472531, 0.36190864, -0.18207191, 0.09431621, -0.03173184, 0.30378476, -0.0680986, -0.22426306, -0.12263116, 0.24350622, 0.02292985, 0.21439615, 0.1712741, -0.3722389, -0.40790412, -0.35991952, -0.3736205, 0.04121229, 0.34964392, 0.34093592, -0.12088361, -0.00075004, -0.16097906, -0.13096544, 0.17002663]
    ], dtype=np.float32),
    np.array([0.0] * 32, dtype=np.float32),
    np.array([[-0.14627874], [-0.22215186], [0.2629217], [-0.31756645], [0.33408672], [0.2284134], [0.00857154], [0.06816602], [-0.2799529], [0.16709483], [0.06370184], [0.04208097], [-0.03786606], [0.38037455], [0.15998226], [0.39232796], [-0.2840253], [0.13968271], [0.04227462], [-0.11299226], [0.0800668], [0.04083297], [-0.22899848], [-0.19703534], [-0.17700578], [-0.02324492], [-0.11840099], [0.40429533], [0.39085168], [-0.0833596], [0.16962683], [-0.01453412]], dtype=np.float32),
    np.array([0.0], dtype=np.float32)
]

def bangun_model_jst(learning_rate=0.001):
    model = Sequential([
        Input(shape=(3,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), # Menggunakan Adam sesuai performa Champion tertinggi
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.set_weights(MODEL_INITIAL_WEIGHTS)
    return model

def jalankan_training_pipeline(db_url):
    logging.info("MEMULAI PIPELINE TRAINING MODEL...")
    print("🔌 Menghubungkan ke Cloud Database untuk menarik data...")
    
    # 1. LOAD DATA FROM DATABASE
    engine = create_engine(db_url)
    
    # Menambahkan ORDER BY untuk menjamin replikasi urutan data 100% sama dengan CSV
    query = "SELECT jenis_kelamin, umur_bulan, tinggi_badan_cm, status_stunting FROM riwayat_pengukuran_balita ORDER BY id_balita ASC"
    df = pd.read_sql(query, con=engine)
    logging.info(f"Berhasil menarik {len(df)} baris data dari database.")

    # 2. FEATURE ENGINEERING (Teks -> Angka)
    df['jenis_kelamin'] = df['jenis_kelamin'].map({'Perempuan': 0, 'Laki-laki': 1})
    df['status_stunting'] = df['status_stunting'].map({'Normal': 0, 'Stunting': 1})
    
    X = df[['jenis_kelamin', 'umur_bulan', 'tinggi_badan_cm']].values
    y = df['status_stunting'].values

    # 3. STRATIFIED DATA SPLITTING (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    logging.info(f"Split selesai. Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. FEATURE SCALING (Anti Data Leakage - R1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simpan scaler untuk keperluan deployment web/api nanti
    os.makedirs('models/export', exist_ok=True)
    joblib.dump(scaler, 'models/export/minmax_scaler.joblib')
    logging.info("MinMaxScaler berhasil disimpan.")

    # 5. HANDLING IMBALANCE DATA (SMOTETomek - Khusus Data Train)
    print("⚖️ Menyeimbangkan distribusi kelas data dengan SMOTETomek...")
    smt = SMOTETomek(random_state=SEED)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train_scaled, y_train)
    logging.info(f"SMOTETomek Selesai. Sebelum: {np.bincount(y_train)} -> Sesudah: {np.bincount(y_train_resampled)}")

    # 6. MODEL TRAINING (Menggunakan Konfigurasi Champion Adam Jurnal)
    print("🧠 Melatih Model Jaringan Saraf Tiruan (JST 3-32-1)...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_resampled, y_train_resampled, test_size=0.20, random_state=SEED, stratify=y_train_resampled
    )

    model = bangun_model_jst(learning_rate=0.001)
    
    checkpoint_path = "models/export/champion_model_jst.keras"
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_accuracy', 
        save_best_only=True, mode='max', verbose=0
    )

    start_time = time.time()
    model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=500, batch_size=32,
        callbacks=[checkpoint], verbose=1
    )
    durasi = time.time() - start_time
    print(f"✅ Training selesai dalam {durasi:.2f} detik!")
    logging.info(f"Model berhasil dilatih dan disimpan di {checkpoint_path}")
    
    # Simpan juga data test untuk dievaluasi pada script berikutnya
    np.savez('data/processed/test_data_eval.npz', X_test=X_test_scaled, y_test=y_test)
    print("💾 Model Champion & Data Evaluasi berhasil dieksport!")

if __name__ == "__main__":
    # ⚠️ Masukkan kembali Connection String Neon.tech kamu di sini
    KONEKSI_DB = "postgresql://neondb_owner:npg_KlTnJ0GFDIh1@ep-bold-fire-ao1htius-pooler.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    
    if KONEKSI_DB == "MASUKKAN_CONNECTION_STRING_NEON_MU_DI_SINI":
        print("❌ Gagal: Isikan terlebih dahulu Connection String Neon.tech kamu pada variabel KONEKSI_DB.")
    else:
        jalankan_training_pipeline(KONEKSI_DB)