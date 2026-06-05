import os
import pandas as pd
import numpy as np
import joblib

def load_best_model():
    """Memuat model terbaik yang sudah disimpan pada Langkah 2"""
    # Kita gunakan Random Forest sebagai model utama karena biasanya performanya paling stabil
    model_path = r"C:\Users\alfin\Eksperimen_SML_MuhamadRizal\models\best_random_forest.pkl"
    
    if not os.path.exists(model_path):
        # Jika Random Forest tidak ada, coba ambil Logistic Regression
        model_path = r"C:\Users\alfin\Eksperimen_SML_MuhamadRizal\models\best_logistic_regression.pkl"
        
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ File model (.pkl) tidak ditemukan di folder 'models'. Pastikan Langkah 2 sudah sukses!")
        
    print(f"✅ Berhasil memuat model dari: {model_path}")
    return joblib.load(model_path)

def data_Inference_simulasi():
    """Membuat data tiruan (1 pasien) untuk simulasi prediksi"""
    # Format kolom harus persis sama dengan fitur yang ada di X_train / X_test
    # Contoh data pasien baru (nilai disesuaikan dengan skala dataset heart disease)
    pasien_baru = {
        'age': [52],
        'sex': [1],        # 1 = Pria, 0 = Wanita
        'cp': [0],         # Tipe nyeri dada (0-3)
        'trestbps': [125], # Tekanan darah resting
        'chol': [212],     # Kolesterol
        'fbs': [0],        # Gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak)
        'restecg': [1],    # Hasil elektrokardiografi resting (0-2)
        'thalach': [168],  # Detak jantung maksimum
        'exang': [0],      # Angina akibat olahraga (1 = ya, 0 = tidak)
        'oldpeak': [1.0],  # Depresi ST yang diinduksi oleh olahraga
        'slope': [2],      # Kemiringan puncak latihan segmen ST (0-2)
        'ca': [2],         # Jumlah pembuluh darah utama (0-3)
        'thal': [3]        # Jenis kelainan darah (0-3)
    }
    return pd.DataFrame(pasien_baru)

if __name__ == "__main__":
    print("================ Menjalankan Tahap Inference ================\n")
    
    # 1. Load Model
    model = load_best_model()
    
    # 2. Ambil data simulasi pasien baru
    data_pasien = data_Inference_simulasi()
    print("\n📊 Data Pasien Baru yang akan Diprediksi:")
    print(data_pasien)
    
    # 3. Lakukan Prediksi
    prediksi = model.predict(data_pasien)
    probabilitas = model.predict_proba(data_pasien)[:, 1] # Probabilitas terkena penyakit jantung
    
    # 4. Tampilkan Hasil Prediksi
    print("\n================ HASIL DIAGNOSIS MODEL ================")
    if prediksi[0] == 1:
        print(f"🚨 STATUS: Terindikasi Penyakit Jantung (Positif)")
    else:
        print(f"✅ STATUS: Jantung Terdeteksi Sehat (Negatif)")
        
    print(f"📈 Tingkat Keyakinan Model: {probabilitas[0]*100:.2f}%")
    print("=======================================================")