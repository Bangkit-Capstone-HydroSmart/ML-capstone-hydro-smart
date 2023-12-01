import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel

# Fungsi untuk membaca data dari file Excel
def read_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    le = LabelEncoder()
    df['tanaman_encoded'] = le.fit_transform(df['Nama'])
    return df

# Fungsi untuk pra-pemrosesan data
def preprocess_data(df):
    # Memisahkan fitur (X) dan target (y)
    X = df[['Luas', 'Suhu', 'PH', 'Kelembapan', 'Penyinaran']]
    y = df['tanaman_encoded']
    
    # Pisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisasi data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Fungsi untuk mendapatkan rekomendasi tanaman
def get_rekomendasi_tanaman(cosine_similarities, df, suhu_pengguna, luas_lahan_pengguna, ph_pengguna, kelembapan_pengguna, penyinaran_pengguna, num_rekomendasi=5):
    idx_tanaman_serupa = df.index[cosine_similarities[:, 0].argsort()[-num_rekomendasi:][::-1]]
    rekomendasi = df.loc[idx_tanaman_serupa, 'Nama'].tolist()
    return list(set(rekomendasi))[:num_rekomendasi]

# Ganti path sesuai dengan lokasi file Excel
excel_path = r'C:\Assigment\dataset2.xlsx'

# Baca dataset dan pra-pemrosesan data
df_tanaman = read_excel_data(excel_path)
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df_tanaman)

# Arsitektur model embeddings
model_tanaman = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(5,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(df_tanaman['Nama'].unique()), activation='softmax')
])

# Kompilasi dan latih model dengan early stopping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_tanaman.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_tanaman.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# Hitung kesamaan kosinus
combined_matrix_tanaman = X_test_scaled
cosine_similarities_tanaman = linear_kernel(combined_matrix_tanaman, combined_matrix_tanaman)

# Contoh penggunaan
suhu_pengguna = 22.0
luas_lahan_pengguna = 80.0
ph_pengguna = 7
kelembapan_pengguna = 70
penyinaran_pengguna = 8

rekomendasi_tanaman = get_rekomendasi_tanaman(cosine_similarities_tanaman, df_tanaman, suhu_pengguna, luas_lahan_pengguna, ph_pengguna, kelembapan_pengguna, penyinaran_pengguna)
print(f"Rekomendasi tanaman untuk suhu {suhu_pengguna}, luas lahan {luas_lahan_pengguna}, PH {ph_pengguna}, kelembapan udara {kelembapan_pengguna}, dan jumlah penyinaran {penyinaran_pengguna}: {rekomendasi_tanaman}")

# Plot grafik loss dan akurasi
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
