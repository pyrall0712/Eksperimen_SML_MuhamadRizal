import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Membaca dataset mentah Iris."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {file_path}")
    return pd.read_csv(file_path)

def pipeline_preprocessing(df):
    """Menjalankan tahapan preprocessing sesuai dengan 6 kolom di screenshot Anda."""
    # 1. Menghapus kolom Id karena tidak dipakai untuk training
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
        
    # 2. Memisahkan Fitur (X) dan Target (y)
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    
    # 3. Standardisasi (Scaling) Fitur Numerik
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 4. Melakukan Train-Test Split (80:20) dengan Stratify agar proporsi spesies seimbang
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Menggabungkan kembali menjadi Dataframe utuh siap pakai
    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    
    return train_df, test_df

def save_processed_data(train_df, test_df, output_dir):
    """Menyimpan hasil preprocessing ke folder tujuan."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'iris_train_processed.csv')
    test_path = os.path.join(output_dir, 'iris_test_processed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Sukses! Data Iris hasil preprocessing disimpan di: {output_dir}")

if __name__ == "__main__":
    # Pastikan nama file CSV dari Kaggle disesuaikan di folder namadataset_raw
    INPUT_FILE = "namadataset_raw/Iris.csv" 
    OUTPUT_DIR = "preprocessing/namadataset_preprocessing"
    
    print("Memulai otomatisasi preprocessing dataset Iris...")
    try:
        raw_data = load_data(INPUT_FILE)
        train_data, test_data = pipeline_preprocessing(raw_data)
        save_processed_data(train_data, test_data, OUTPUT_DIR)
    except Exception as e:
        print(f"Terjadi kesalahan saat preprocessing: {e}")