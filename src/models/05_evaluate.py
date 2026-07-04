import os
import numpy as np
import logging
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def evaluasi_model():
    print("📊 Memulai Proses Evaluasi Model...")
    
    # 1. Load Data Test (Data yang tidak pernah dilihat model saat training)
    test_data_path = os.path.join('data', 'processed', 'test_data_eval.npz')
    if not os.path.exists(test_data_path):
        print("❌ File test_data_eval.npz tidak ditemukan. Pastikan training selesai dulu!")
        return
        
    data = np.load(test_data_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # 2. Load Model Champion
    model_path = os.path.join('models', 'export', 'champion_model_jst.keras')
    if not os.path.exists(model_path):
        print("❌ Model champion_model_jst.keras tidak ditemukan!")
        return
        
    model = load_model(model_path)
    logging.info("Model dan Data Evaluasi berhasil dimuat.")

    # 3. Lakukan Prediksi
    # Karena output layer pakai Sigmoid, hasilnya probabilitas (0.0 - 1.0)
    y_prob = model.predict(X_test, verbose=0).flatten()
    # Ubah probabilitas jadi kelas biner dengan threshold 0.5
    y_pred = (y_prob > 0.5).astype(int)

    # 4. Hitung Metrik Evaluasi
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0

    print("\n" + "="*45)
    print("🏆 HASIL EVALUASI MODEL CHAMPION JST 🏆")
    print("="*45)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print("="*45)

    # 5. Plot & Simpan Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    
    # Menggunakan style dari seaborn agar rapi seperti di Colab
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Stunting'], 
                yticklabels=['Normal', 'Stunting'],
                annot_kws={'size': 14})
                
    plt.xlabel('Prediksi', fontsize=12)
    plt.ylabel('Aktual', fontsize=12)
    plt.title('Confusion Matrix - Model Prediksi Stunting', fontsize=14, fontweight='bold')
    
    # Buat folder laporan jika belum ada
    os.makedirs(os.path.join('reports', 'figures'), exist_ok=True)
    fig_path = os.path.join('reports', 'figures', 'confusion_matrix.png')
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"\n🎨 Visualisasi Confusion Matrix berhasil disimpan di: {fig_path}")

if __name__ == "__main__":
    evaluasi_model()