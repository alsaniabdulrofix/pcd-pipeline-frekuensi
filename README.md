# Pipeline Pengolahan Citra Digital Berbasis Transformasi Frekuensi

Repositori ini berisi implementasi *mini project* untuk mata kuliah Pengolahan Citra Digital. 
Proyek ini mengusulkan rancang bangun arsitektur *pipeline* untuk memproses citra digital pada ranah frekuensi guna melakukan ekstraksi fitur visual.

## 📌 Alur Pemrosesan (Pipeline)
Sistem ini dibangun secara prosedural dan modular dengan tahapan berikut:
1. **Prapemrosesan:** Membaca citra, konversi ke *grayscale*, dan normalisasi resolusi (512x512).
2. **Transformasi:** Mengubah ranah spasial ke ranah frekuensi menggunakan algoritma *Fast Fourier Transform* (FFT).
3. **Konstruksi Filter:** Pembuatan *Ideal Low Pass Filter* (LPF) dan *Ideal High Pass Filter* (HPF).
4. **Filtrasi & Rekonstruksi:** Penerapan matriks *masking* dan pengembalian spektrum ke wujud citra menggunakan *Inverse Fast Fourier Transform* (IFFT).
5. **Visualisasi:** Menampilkan perbandingan hasil ekstraksi secara berdampingan.

## 🚀 Hasil Pengujian
Sistem mampu melakukan dekomposisi fitur secara paralel:
* **Low Pass Filter (LPF):** Berhasil meredam frekuensi tinggi untuk mereduksi *noise* dan menghaluskan tekstur citra.
* **High Pass Filter (HPF):** Berhasil meredam frekuensi rendah untuk mengekstraksi kerangka garis tepi (*edge detection*) dari latar belakang.

## 🛠️ Cara Menjalankan Program
Pastikan telah menginstal pustaka yang dibutuhkan:

`pip install opencv-python numpy matplotlib`


Jalankan perintah berikut di terminal:
`python main_pipeline.py`

## Cara install Virtual Environtment

'python -m venv env'

## Aktifkan Virtual Environment

Untuk Windows (Command Prompt / CMD):
'env\Scripts\activate'

Untuk Windows (PowerShell):
'.\env\Scripts\Activate.ps1'

Untuk macOS / Linux:
'source env/bin/activate'