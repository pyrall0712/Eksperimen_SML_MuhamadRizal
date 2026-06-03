import os
import pandas as pd
from sklearn.model_selection import train_test_split

def run_automated_preprocessing():
    print("=== Memulai Proses Otomatisasi Preprocessing ===")
    
    # 1. Definisikan Jalur Berkas (Path) secara Dinamis
    # Menggunakan basePath agar aman saat dieksekusi di komputer lokal maupun GitHub Actions Runner
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Menyesuaikan dengan nama folder di screenshot panduan Anda
    raw_data_dir = os.path.join(base_dir, 'namadataset_raw')
    output_dir = os.path.join(base_dir, 'preprocessing', 'namadataset_preprocessing')
    
    # Pastikan folder output dibuat otomatis jika belum ada di server GitHub
    os.makedirs(output_dir, exist_ok=True)
    
    # Ambil berkas CSV pertama yang ada di dalam folder namadataset_raw
    try:
        raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        if not raw_files:
            raise FileNotFoundError("Tidak ditemukan berkas CSV di dalam folder 'namadataset_raw'!")
        
        raw_file_path = os.path.join(raw_data_dir, raw_files[0])
        print(f"Membaca data mentah dari: {raw_file_path}")
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        print(f"Error saat membaca data mentah: {str(e)}")
        return

    # 2. Proses Pembersihan Data (Handling Missing Values / NaN)
    print("Membersihkan nilai kosong (NaN)...")
    # Mengisi nilai NaN pada kolom numerik dengan nilai rata-rata (mean)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Jika ada kolom teks yang kosong, isi dengan modus atau drop
    df = df.dropna()

    # 3. Pembagian Data (Data Splitting) - Syarat Utama Kriteria 1
    print("Melakukan pembagian data menjadi Train dan Test (Rasio 80:20)...")
    train_df, test_df = train_test_split(df, test_split=0.2, random_state=42)

    # 4. Menyimpan Hasil Ekstraksi ke Folder Preprocessing
    train_output_path = os.path.join(output_dir, 'iris_train_processed.csv')
    test_output_path = os.path.join(output_dir, 'iris_test_processed.csv')
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"🎉 Sukses menyimpan data train ke: {train_output_path}")
    print(f"🎉 Sukses menyimpan data test ke: {test_output_path}")
    print("=== Proses Preprocessing Selesai ===")

if __name__ == "__main__":
    run_automated_preprocessing()