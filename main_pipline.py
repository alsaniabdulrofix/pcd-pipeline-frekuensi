import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# TAHAP 1: MODUL PREPROCESSING
# ==========================================
def preprocess_image(image_path):
    """Membaca citra, konversi ke grayscale, dan normalisasi dimensi (512x512)."""
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError(f"Galat: Citra '{image_path}' tidak ditemukan di direktori.")
    
    img_resized = cv2.resize(img, (512, 512))
    return img_resized


# ==========================================
# TAHAP 2: MODUL TRANSFORMASI SPEKTRUM (FFT)
# ==========================================
def transform_to_frequency(img):
    """Konversi ranah spasial ke ranah frekuensi beserta pergeseran ke pusat matriks."""
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    return f_shift


def create_lpf_mask(shape, radius=60):
    """Konstruksi Ideal Low Pass Filter (mempertahankan frekuensi rendah/warna dasar)."""
    baris, kolom = shape
    pusat_baris, pusat_kolom = baris // 2, kolom // 2
    
    mask = np.zeros((baris, kolom), np.uint8)
    mask[pusat_baris - radius : pusat_baris + radius, 
         pusat_kolom - radius : pusat_kolom + radius] = 1
    return mask

def create_hpf_mask(shape, radius=60):
    """Konstruksi Ideal High Pass Filter (mempertahankan frekuensi tinggi/tepi objek)."""
    baris, kolom = shape
    pusat_baris, pusat_kolom = baris // 2, kolom // 2
    
    mask = np.ones((baris, kolom), np.uint8)
    mask[pusat_baris - radius : pusat_baris + radius, 
         pusat_kolom - radius : pusat_kolom + radius] = 0
    return mask
