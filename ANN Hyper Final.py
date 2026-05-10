import pandas as pd
import numpy as np

# Path dataset
input_file = 'dataset5000final.csv'
output_file = 'dataset5000final_processed.csv'

# Load dataset
df = pd.read_csv(input_file)

# Daftar kolom wajib
required_columns = [
    'kategori_produk', 'review', 'family_template', 'jenis_produk', 'merek',
    'warna', 'material', 'bentuk_kemasan', 'log_harga', 'is_non_brand',
    'berat_kg', 'tebal_mm', 'jumlah_lembar', 'ukuran_cm', 'panjang_meter', 'yard',
    'box_panjang_cm', 'box_lebar_cm', 'box_tinggi_cm', 'log_volume_box_cm3',
    'lebar_cm_roll', 'panjang_meter_roll', 'lebar_pair_yard', 'panjang_yard_pair',
    'is_bflute', 'is_cflute', 'is_foam', 'is_kraft',
    'jumlah_dimensi_x', 'panjang_nama', 'jumlah_token', 'nama_produk_normalized'
]

# Tambahkan kolom yang hilang dengan nilai default
for col in required_columns:
    if col not in df.columns:
        if col == 'review':
            df[col] = 1.0  # default review minimal
        elif col in ['kategori_produk', 'family_template', 'jenis_produk', 'merek', 'warna', 'material', 'bentuk_kemasan', 'nama_produk_normalized']:
            df[col] = 'lainnya'
        else:
            df[col] = 0.0

# Pastikan tipe data
numeric_cols = [
    'log_harga', 'is_non_brand', 'berat_kg', 'tebal_mm', 'jumlah_lembar', 'ukuran_cm',
    'panjang_meter', 'yard', 'box_panjang_cm', 'box_lebar_cm', 'box_tinggi_cm', 'log_volume_box_cm3',
    'lebar_cm_roll', 'panjang_meter_roll', 'lebar_pair_yard', 'panjang_yard_pair',
    'jumlah_dimensi_x', 'panjang_nama', 'jumlah_token', 'is_bflute', 'is_cflute', 'is_foam', 'is_kraft'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# Kolom kategori
categorical_cols = ['kategori_produk', 'family_template', 'jenis_produk', 'merek', 'warna', 'material', 'bentuk_kemasan', 'nama_produk_normalized']
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna('lainnya')

# Review harus antara 1.0-5.0
df['review'] = pd.to_numeric(df['review'], errors='coerce')
df['review'] = df['review'].clip(1.0, 5.0).fillna(1.0)

# Kolom split_grouped opsional
df['split_grouped'] = df.get('split_grouped', 'unknown')
df['split_grouped'] = df['split_grouped'].astype(str).str.lower().fillna('unknown')

# Reset index dan simpan
df.reset_index(drop=True, inplace=True)
df.to_csv(output_file, index=False)
print(f"Dataset sudah diproses dan disimpan di: {output_file}")
