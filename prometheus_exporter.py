import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from prometheus_client import Counter, make_asgi_app
import pickle
import os
import numpy as np

app = FastAPI(title="Iris Model Serving with Prometheus Monitoring")

# 1. Inisialisasi Metrik Prometheus
PREDICTION_COUNTER = Counter(
    "iris_predictions_total", 
    "Total number of Iris predictions made by the model",
    ["species_predicted"]
)

# 2. Struktur Data Input untuk API
class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

# 3. Membaca model.pkl dan validasi objek
model_path = "deployment/model.pkl"
model = None

if os.path.exists(model_path):
    print(f"📂 Membuka berkas model dari: {model_path}")
    with open(model_path, "rb") as f:
        loaded_object = pickle.load(f)
    
    # Validasi apakah objek memiliki fungsi 'predict'
    if hasattr(loaded_object, "predict"):
        print("✅ Berhasil memuat model Machine Learning utama!")
        model = loaded_object
    else:
        print("⚠️ Berkas pkl ditemukan, tetapi isinya bukan model (numpy array/data). Mengaktifkan model cadangan...")

if model is None:
    # Membuat model cadangan otomatis agar program tidak crash
    from sklearn.datasets import load_iris
    iris = load_iris()
    model = RandomForestClassifier(random_state=42)
    model.fit(iris.data, iris.target)
    print("✅ Model cadangan aktif dan siap menerima data.")

@app.post("/predict")
def predict(data: IrisInput):
    input_data = pd.DataFrame([{
        "SepalLengthCm": data.SepalLengthCm,
        "SepalWidthCm": data.SepalWidthCm,
        "PetalLengthCm": data.PetalLengthCm,
        "PetalWidthCm": data.PetalWidthCm
    }])
    
    # Eksekusi prediksi
    prediction = model.predict(input_data)[0]
    
    # Mapping hasil prediksi ke nama spesies
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    result_species = species_map.get(prediction, str(prediction))
    
    # Naikkan angka counter Prometheus
    PREDICTION_COUNTER.labels(species_predicted=result_species).inc()
    
    return {"prediction": result_species}

# 4. Endpoint '/metrics'
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    print("🚀 Memulai Server Serving di http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)