"""
Konfigurasi Sistem Deteksi Hand Sign
Jaringan Syaraf Tiruan Backpropagation

File ini berisi seluruh parameter yang digunakan sistem, mulai dari
path file, arsitektur jaringan, hyperparameter training, hingga
pengaturan kamera dan tampilan. Ubah nilai di sini untuk menyesuaikan
eksperimen tanpa perlu menyentuh kode utama.
"""

import os

# ======================================================================
# Path File dan Direktori
# ======================================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
DATASET_CSV  = os.path.join(DATASET_DIR, "landmarks.csv")
MODEL_PATH   = os.path.join(MODEL_DIR, "jst_hand_sign_model.npz")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.npz")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")


# ======================================================================
# Label Kelas (Angka 0 sampai 9)
# ======================================================================

NUM_CLASSES  = 10
LABELS       = {i: str(i) for i in range(10)}
LABEL_NAMES  = {i: f"Angka {i}" for i in range(10)}


# ======================================================================
# Konfigurasi MediaPipe Hand Detection
# ======================================================================

MAX_NUM_HANDS           = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.5
NUM_LANDMARKS = 21      # MediaPipe mendeteksi 21 titik pada tangan
NUM_COORDS    = 3       # Setiap titik memiliki koordinat x, y, z


# ======================================================================
# Arsitektur Jaringan Syaraf Tiruan (JST)
# ======================================================================
#
# Komposisi fitur input (total 86 fitur):
#   63  = koordinat landmark (21 titik x 3 koordinat)
#   5   = sudut tekukan tiap jari
#   4   = jarak ujung jari ke ibu jari
#   5   = status biner lurus/terlipat per jari
#   5   = status biner jari ke atas (is_up)
#   4   = jarak antar ujung jari yang berdekatan

INPUT_SIZE  = 86
HIDDEN_LAYERS = [256, 128, 64]
OUTPUT_SIZE = NUM_CLASSES

# Arsitektur lengkap: [86, 256, 128, 64, 10]
NETWORK_ARCHITECTURE = [INPUT_SIZE] + HIDDEN_LAYERS + [OUTPUT_SIZE]


# ======================================================================
# Hyperparameter Training
# ======================================================================

LEARNING_RATE          = 0.001     # Laju pembelajaran awal
EPOCHS                 = 600       # Jumlah epoch maksimal
BATCH_SIZE             = 32        # Ukuran mini-batch
MOMENTUM               = 0.9       # Koefisien momentum SGD
LEARNING_RATE_DECAY    = 0.999     # Faktor peluruhan learning rate per epoch
EARLY_STOPPING_PATIENCE = 80       # Epoch toleransi tanpa perbaikan sebelum berhenti
REGULARIZATION_LAMBDA  = 0.0003    # Bobot L2 regularization


# ======================================================================
# Konfigurasi Dataset
# ======================================================================

TEST_SPLIT            = 0.2     # Proporsi data untuk pengujian akhir
VALIDATION_SPLIT      = 0.1     # Proporsi data untuk validasi saat training
MIN_SAMPLES_PER_CLASS = 30      # Jumlah minimum sampel yang disarankan per kelas
RANDOM_SEED           = 42      # Seed acak untuk hasil yang dapat direproduksi


# ======================================================================
# Konfigurasi Kamera
# ======================================================================

CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX  = 0       # Indeks kamera (0 = webcam bawaan)
FPS_DISPLAY   = True


# ======================================================================
# Fitur Suara (Text-to-Speech)
# ======================================================================

VOICE_ENABLED  = True
VOICE_RATE     = 150    # Kecepatan bicara (kata per menit)
VOICE_COOLDOWN = 2.0    # Jeda minimum antar pengucapan (detik)
VOICE_VOLUME   = 1.0    # Volume suara (rentang 0.0 hingga 1.0)


# ======================================================================
# Warna UI (format BGR untuk OpenCV)
# ======================================================================

COLOR_GREEN     = (0, 255, 0)
COLOR_RED       = (0, 0, 255)
COLOR_BLUE      = (255, 0, 0)
COLOR_YELLOW    = (0, 255, 255)
COLOR_WHITE     = (255, 255, 255)
COLOR_BLACK     = (0, 0, 0)
COLOR_ORANGE    = (0, 165, 255)
COLOR_CYAN      = (255, 255, 0)
COLOR_PANEL_BG  = (40, 40, 40)
COLOR_RECORDING = (0, 0, 255)
COLOR_READY     = (0, 200, 0)


# ======================================================================
# Threshold Deteksi
# ======================================================================

# Nilai confidence minimum agar prediksi dianggap valid dan suara dibunyikan.
# Diturunkan sedikit agar kelas yang lebih sulit seperti angka 4 tetap terdeteksi.
PREDICTION_CONFIDENCE_THRESHOLD = 0.45
