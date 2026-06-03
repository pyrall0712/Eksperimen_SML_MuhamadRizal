import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import mlflow

def train_tuning():
    # ==============================================================================
    # KONFIGURASI DAGSHUB (Wajib diisi sesuai akun Anda)
    # ==============================================================================
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'pyrall0712'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'b43fc29f9e672cd28aeb14a3a9690348f1cf146a'
    mlflow.set_tracking_uri('https://dagshub.com/pyrall0712/Eksperimen_SML_MuhamadRizal.mlflow')

    # 1. Membaca data Iris hasil preprocessing dari Kriteria 1
    train_df = pd.read_csv('preprocessing/namadataset_preprocessing/iris_train_processed.csv')
    test_df = pd.read_csv('preprocessing/namadataset_preprocessing/iris_test_processed.csv')

    # Membersihkan data dari nilai kosong (NaN) agar aman saat training
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # 2. Memisahkan Fitur dan Target (Species)
    X_train = train_df.drop(columns=['Species'])
    y_train = train_df['Species']
    X_test = test_df.drop(columns=['Species'])
    y_test = test_df['Species']

    # 3. Mengatur Nama Eksperimen di MLflow (Gunakan nama yang berbeda dari Baseline)
    mlflow.set_experiment("Iris_Classification_Tuning")

    # 4. Menentukan Variasi Hyperparameter yang Akan Diuji (Grid Search)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5]
    }

    # 5. Inisialisasi Model Dasar dan Penguji Kombinasi (GridSearchCV)
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy')
    
    print("Sedang mencari kombinasi hyperparameter terbaik...")
    grid_search.fit(X_train, y_train)

    # Menangkap model dan parameter terbaik hasil pencarian
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 6. Memulai Recording Eksperimen Terbaik ke DagsHub
    with mlflow.start_run(run_name="Random_Forest_Tuned"):
        
        # Evaluasi Akurasi Model Terbaik menggunakan Data Test
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # MENULIS LOG PARAMETER TERBAIK KE MLFLOW
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        # MENULIS LOG METRIC AKURASI KE MLFLOW
        mlflow.log_metric("accuracy", acc)

        print(f"\nBerhasil! Akurasi Model Hasil Tuning: {acc:.4f}")
        print(f"Parameter Terbaik: {best_params}")

if __name__ == "__main__":
    train_tuning()