import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. Mengatur lokasi penyimpanan MLflow lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart_Disease_Eksperimen_Rizal")

def load_data():
    """Memuat data hasil preprocessing berdasarkan gambar struktur folder"""
    base_dir = r"C:\Users\alfin\Eksperimen_SML_MuhamadRizal"
    
    # Menyesuaikan dengan struktur: preprocessing/namadataset_preprocessing
    data_dir = os.path.join(base_dir, "preprocessing", "namadataset_preprocessing")
    
    print(f"🔄 Membaca data bersih dari: {data_dir}")
        
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_track_model(model_name, base_model, param_grid, X_train, X_test, y_train, y_test):
    """Melatih model dengan GridSearchCV dan mencatat hasilnya ke MLflow"""
    print(f"\n================ Melatih Model: {model_name} ================")
    
    # Hyperparameter Tuning menggunakan GridSearchCV
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Prediksi menggunakan model terbaik
    y_pred = best_model.predict(X_test)
    
    # Hitung Metrik Evaluasi
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    print(f"Parameter Terbaik: {best_params}")
    print(f"Metrik Evaluasi -> Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_score']:.4f}")
    
    # Mencatat eksperimen ke dalam MLflow Run
    with mlflow.start_run(run_name=model_name):
        # Log Hyperparameters
        for param_name, param_val in best_params.items():
            mlflow.log_param(param_name, param_val)
        
        # Log Metrics
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)
            
        # Log Model ke MLflow Artifacts
        mlflow.sklearn.log_model(best_model, artifact_path=model_name)
        
    # Simpan model terbaik secara fisik ke folder lokal untuk Kriteria 3 nanti
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/best_{model_name.lower().replace(' ', '_')}.pkl")
    print(f"✅ Model {model_name} berhasil disimpan lokal dan dilacak di MLflow!")

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # --- MODEL 1: RANDOM FOREST ---
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'random_state': [42]
    }
    train_and_track_model("Random Forest", RandomForestClassifier(), rf_params, X_train, X_test, y_train, y_test)
    
    # --- MODEL 2: LOGISTIC REGRESSION ---
    lr_params = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [100, 500],
        'random_state': [42]
    }
    train_and_track_model("Logistic Regression", LogisticRegression(), lr_params, X_train, X_test, y_train, y_test)
    
    print("\n🎉 Semua eksperimen model berhasil dijalankan dan tercatat di MLflow!")