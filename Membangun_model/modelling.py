import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# ==============================================================================
# KONFIGURASI DAGSHUB (Wajib diisi sesuai akun Anda)
# ==============================================================================
os.environ['MLFLOW_TRACKING_USERNAME'] = 'pyrall0712'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'b43fc29f9e672cd28aeb14a3a9690348f1cf146a'
mlflow.set_tracking_uri('https://dagshub.com/pyrall0712/Eksperimen_SML_MuhamadRizal.mlflow')

def train_baseline():
    # 1. Membaca data Iris hasil preprocessing dari Kriteria 1
    train_df = pd.read_csv('preprocessing/namadataset_preprocessing/iris_train_processed.csv')
    test_df = pd.read_csv('preprocessing/namadataset_preprocessing/iris_test_processed.csv')
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    # 2. Memisahkan Fitur dan Target (Species)
    X_train = train_df.drop(columns=['Species'])
    y_train = train_df['Species']
    X_test = test_df.drop(columns=['Species'])
    y_test = test_df['Species']
    
    # 3. Mengatur Nama Eksperimen di MLflow
    mlflow.set_experiment("Iris_Classification_Baseline")
    
    # 4. Memulai Recording Eksperimen ke DagsHub
    with mlflow.start_run(run_name="CI_Automated_Run"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Logging parameter dan metrik
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        # BARIS PERBAIKAN: Paksa sklaern untuk mencatat model ke sub-folder "model"
        # Menambahkan input_example atau signature opsional agar MLflow mencatatnya sebagai model valid
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model",
            registered_model_name="Iris_RandomForest_Model"
        )
        
        print(f"Berhasil! Akurasi Model: {acc:.4f}")

if __name__ == "__main__":
    train_baseline()