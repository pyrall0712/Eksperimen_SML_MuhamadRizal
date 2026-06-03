import requests

# URL Endpoint API lokal Flask Anda
url = 'http://127.0.0.1:5000/predict'

# Contoh data fitur Iris untuk ditest (sesuaikan nama kolomnya dengan data Anda jika perlu)
# Di bawah ini adalah contoh struktur data JSON yang dikirimkan
data_input = {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}

print("Mengirim data uji coba ke API...")
try:
    response = requests.post(url, json=data_input)
    print("\n--- HASIL RESPON API ---")
    print(response.json())
except Exception as e:
    print(f"Gagal terhubung ke API: {e}")