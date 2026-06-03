import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def export_best_model():
    print("Membaca data untuk export model...")
    # 1. Membaca data Iris hasil preprocessing
    train_df = pd.read_csv('preprocessing/namadataset_preprocessing/iris_train_processed.csv')
    train_df = train_df.dropna()

    X_train = train_df.drop(columns=['Species'])
    y_train = train_df['Species']

    # 2. Inisialisasi model dengan Parameter Terbaik hasil tuning Anda kemarin
    print("Melatih model dengan konfigurasi hyperparameter terbaik...")
    best_model = RandomForestClassifier(
        max_depth=None, 
        min_samples_split=2, 
        n_estimators=200, 
        random_state=42
    )
    best_model.fit(X_train, y_train)

    # 3. Membuat folder 'deployment' jika belum ada
    os.makedirs('deployment', exist_ok=True)

    # 4. Menyimpan model ke dalam folder deployment
    model_path = 'deployment/model.pkl'
    joblib.dump(best_model, model_path)
    print(f"🎉 Sukses! File model berhasil disimpan di: {model_path}")

if __name__ == "__main__":
    export_best_model()