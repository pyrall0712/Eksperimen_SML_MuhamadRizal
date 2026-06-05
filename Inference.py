import requests
import time
import random

url = "http://127.0.0.1:8000/predict"

print("🎯 Memulai simulasi request model otomatis (Tekan Ctrl+C untuk berhenti)...")

while True:
    # Membuat data acak mirip skala bunga iris asli
    payload = {
        "SepalLengthCm": round(random.uniform(4.3, 7.9), 1),
        "SepalWidthCm": round(random.uniform(2.0, 4.4), 1),
        "PetalLengthCm": round(random.uniform(1.0, 6.9), 1),
        "PetalWidthCm": round(random.uniform(0.1, 2.5), 1)
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"🔹 Mengirim data -> Hasil prediksi server: {response.json()['prediction']}")
    except Exception as e:
        print("❌ Gagal terhubung ke server. Pastikan prometheus_exporter.py sudah berjalan di terminal lain!")
        
    # Beri jeda 1 detik sebelum mengirim data berikutnya
    time.sleep(1)