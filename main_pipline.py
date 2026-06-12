import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# TAHAP 1: MODUL PREPROCESSING
# ==========================================
def preprocess_image(image_path):
    
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError(f"Galat: Citra '{image_path}' tidak ditemukan di direktori.")
    
    img_resized = cv2.resize(img, (512, 512))
    return img_resized


# ==========================================
# TAHAP 2: MODUL TRANSFORMASI SPEKTRUM (FFT)
# ==========================================
def transform_to_frequency(img):
   
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    return f_shift


# ==========================================
# TAHAP 3: MODUL KONSTRUKSI FILTER MASKING
# ==========================================
def create_lpf_mask(shape, radius=60):
    
    baris, kolom = shape
    pusat_baris, pusat_kolom = baris // 2, kolom // 2
    
    mask = np.zeros((baris, kolom), np.uint8)
    mask[pusat_baris - radius : pusat_baris + radius, 
         pusat_kolom - radius : pusat_kolom + radius] = 1
    return mask

def create_hpf_mask(shape, radius=60):
    
    baris, kolom = shape
    pusat_baris, pusat_kolom = baris // 2, kolom // 2
    
    mask = np.ones((baris, kolom), np.uint8)
    mask[pusat_baris - radius : pusat_baris + radius, 
         pusat_kolom - radius : pusat_kolom + radius] = 0
    return mask


def apply_mask_and_inverse(f_shift, mask):
    
    f_filtered = f_shift * mask
    
    f_ishift = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(f_ishift)
    
    img_back = np.abs(img_back)
    return img_back
