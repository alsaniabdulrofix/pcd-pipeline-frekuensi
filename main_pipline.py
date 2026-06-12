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

# ==========================================
# TAHAP 4: MODUL FILTRASI DAN INVERSI (IFFT)
# ==========================================
def apply_mask_and_inverse(f_shift, mask):
    
    f_filtered = f_shift * mask
    
    f_ishift = np.fft.ifftshift(f_filtered)
    img_back = np.fft.ifft2(f_ishift)
    
    img_back = np.abs(img_back)
    return img_back


# ==========================================
# TAHAP 5: ARSITEKTUR PIPELINE UTAMA
# ==========================================
def run_pipeline(image_path):
    print("[INFO] Memulai eksekusi pipeline pengolahan citra...")
    
    img = preprocess_image(image_path)
    f_shift = transform_to_frequency(img)
    
    mask_lpf = create_lpf_mask(img.shape, radius=60)
    mask_hpf = create_hpf_mask(img.shape, radius=60)
    
    hasil_lpf = apply_mask_and_inverse(f_shift, mask_lpf)
    hasil_hpf = apply_mask_and_inverse(f_shift, mask_hpf)
    
    print("[INFO] Membuka jendela visualisasi...")
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Citra Prapemrosesan (Spasial)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(hasil_lpf, cmap='gray')
    plt.title('Ekstraksi Fitur Halus (LPF)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hasil_hpf, cmap='gray')
    plt.title('Ekstraksi Garis Tepi (HPF)')
    plt.axis('off')

    plt.tight_layout(pad=3.0) 
    plt.show()

if __name__ == "__main__":
    path_gambar = "vespa.jpg" 
    try:
        run_pipeline(path_gambar)
    except Exception as err:
        print(f"[ERROR] Eksekusi terhenti: {err}")
