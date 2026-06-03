from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Memuat model yang sudah diexport sebelumnya
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "success",
        "message": "API Deployment Model Iris Berhasil Berjalan!"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data JSON yang dikirim oleh pengguna
        data = request.get_json()
        
        # Mengubah data input menjadi DataFrame agar sesuai dengan format model
        input_data = pd.DataFrame([data])
        
        # Melakukan prediksi menggunakan model
        prediction = model.predict(input_data)[0]
        
        # Mengembalikan hasil prediksi dalam bentuk JSON
        return jsonify({
            "status": "success",
            "prediction": str(prediction)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal melakukan prediksi: {str(e)}"
        })

if __name__ == '__main__':
    # Berjalan di lokal port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)