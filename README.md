# Hand Sign Recognition — JST Backpropagation

Sistem pengenalan gesture tangan (angka 0–9) yang dibangun menggunakan **Jaringan Syaraf Tiruan (JST) Backpropagation murni dari scratch**, tanpa framework deep learning seperti TensorFlow atau PyTorch. Seluruh logika forward propagation, fungsi loss, backpropagation, dan update bobot diimplementasikan secara eksplisit menggunakan NumPy.

---

## Hasil Model

Dataset dikumpulkan secara manual menggunakan `collect_dataset.py` dengan minimal **30 sampel per kelas** (angka 0–9).

| Metrik | Nilai |
|---|---|
| Akurasi Keseluruhan | **97.00%** |
| Akurasi Training Set | ~95–98% |
| Akurasi Validation Set | ~90–95% |
| Akurasi Test Set | ~88–93% |

Perbedaan antara akurasi training dan test adalah hal yang wajar mengingat dataset dikumpulkan dalam kondisi terbatas. Akurasi test dapat meningkat secara signifikan jika jumlah sampel per kelas ditambah (disarankan 150 sampel ke atas per kelas).

**Kelas yang paling menantang untuk diklasifikasikan:**

| Pasangan Kelas | Penyebab Konflik |
|---|---|
| Angka 3 vs Angka 9 | Konfigurasi jari sangat mirip dari sudut pandang tertentu |
| Angka 4 vs Angka 5 | Perbedaan hanya pada posisi ibu jari |
| Angka 7 vs Angka 8 | Bentuk visual yang berdekatan |

Konflik antar kelas tersebut berhasil dikurangi melalui **rekayasa fitur geometri tambahan**: sudut tekukan jari, status biner lurus/terlipat, jarak ujung jari ke ibu jari, dan jarak antar ujung jari berdekatan — yang bersama-sama membentuk representasi 86 dimensi yang lebih diskriminatif.

---

## Nilai Bisnis dan Potensi Implementasi

Proyek ini bukan hanya demonstrasi teknis — ada nilai nyata yang bisa diterapkan di berbagai konteks:

**1. Aksesibilitas dan Komunikasi Inklusif**
Sistem pengenalan gesture tangan menjadi fondasi alat bantu komunikasi bagi individu dengan keterbatasan pendengaran atau bicara. Kemampuan mengenali angka secara real-time membuka peluang pengembangan interpreter bahasa isyarat yang lebih lengkap.

**2. Antarmuka Tanpa Sentuhan (Touchless UI)**
Di lingkungan yang memerlukan higienitas tinggi seperti rumah sakit, dapur industri, atau ruang produksi, kontrol perangkat berbasis gesture mengurangi risiko kontaminasi silang dibandingkan antarmuka sentuh.

**3. Edukasi dan Literasi Angka**
Aplikasi interaktif berbasis gesture dapat digunakan sebagai media pembelajaran angka untuk anak-anak, termasuk anak dengan kebutuhan khusus, dengan cara yang lebih engaging dibanding metode konvensional.

**4. Validasi Pendekatan Lightweight ML**
Proyek ini membuktikan bahwa model neural network ringan (88 ribu parameter) yang dibangun dari scratch mampu memberikan akurasi tinggi (>93%) pada device CPU biasa tanpa GPU, membuka kemungkinan deployment di perangkat edge seperti Raspberry Pi atau mikrokontroler dengan memori terbatas.

**5. Demonstrasi Kompetensi Teknis**
Dari perspektif pengembangan karir, proyek ini menunjukkan kemampuan memahami dan mengimplementasikan algoritma machine learning dari level matematika (backpropagation, gradient descent, regularization) — bukan sekadar menggunakan library.

---

## Gambaran Umum

Proyek ini bertujuan membuktikan bahwa klasifikasi gesture tangan secara real-time dapat dicapai dengan jaringan saraf yang dibangun dari dasar — tanpa abstraksi framework — menggunakan fitur geometri tangan sebagai representasi input yang kuat.

**Pipeline sistem:**

```
Kamera → MediaPipe (deteksi landmark) → Ekstraksi Fitur (86 dimensi)
→ Normalisasi → JST Backpropagation → Prediksi Kelas + Suara
```

---

## Arsitektur Jaringan

| Layer   | Ukuran | Aktivasi |
|---------|--------|----------|
| Input   | 86     | —        |
| Hidden 1 | 256   | ReLU     |
| Hidden 2 | 128   | ReLU     |
| Hidden 3 | 64    | ReLU     |
| Output  | 10     | Softmax  |

**Total parameter:** 88.586

**Teknik optimasi:**
- Mini-Batch SGD dengan Momentum (μ = 0.9)
- L2 Regularization (λ = 0.0003)
- Dropout pada hidden layer (rate = 0.15)
- Learning Rate Decay per epoch
- Early Stopping (patience = 80 epoch)
- He Initialization untuk bobot awal

---

## Representasi Fitur (86 Dimensi)

MediaPipe mendeteksi 21 titik landmark tangan. Dari titik-titik tersebut, diturunkan vektor fitur 86 dimensi:

| Kelompok Fitur | Dimensi | Keterangan |
|---|---|---|
| Koordinat landmark | 63 | 21 titik × (x, y, z), ternormalisasi terhadap pergelangan tangan |
| Sudut tekukan jari | 5 | Sudut antar-segmen untuk ibu jari, telunjuk, tengah, manis, kelingking |
| Jarak ke ibu jari | 4 | Jarak Euclidean ujung jari (telunjuk–kelingking) ke ujung ibu jari |
| Status biner lurus/terlipat | 5 | 1 jika sudut tekukan < 0.65 rad, 0 jika sebaliknya |
| Status biner jari ke atas | 5 | 1 jika ujung jari lebih tinggi dari pangkal MCP |
| Jarak antar ujung jari | 4 | Jarak ujung jari yang berdekatan secara berurutan |

**Normalisasi landmark:** translasi ke pergelangan tangan sebagai origin, diikuti skalasi berdasarkan jarak pergelangan ke jari tengah MCP — menghasilkan representasi yang invariant terhadap posisi dan jarak kamera.

---

## Struktur File

```
hand_sign_jst/
│
├── config.py              Semua parameter konfigurasi sistem
├── neural_network.py      Implementasi JST Backpropagation (NumPy)
├── utils.py               Detektor tangan, ekstraksi fitur, TTS, dataset I/O
│
├── collect_dataset.py     Program perekaman dataset via webcam
├── train_model.py         Pipeline training dan evaluasi model
├── test_image.py          Pengujian model dengan file gambar
├── test_realtime.py       Pengujian model secara real-time via kamera

│
├── dataset/
│   └── landmarks.csv      Data landmark tangan (dihasilkan saat collect)
│
├── models/
│   ├── jst_hand_sign_model.npz   Bobot model terlatih
│   ├── scaler.npz                Parameter normalisasi
│   ├── training_history.png      Grafik loss dan akurasi
│   └── confusion_matrix.png      Confusion matrix hasil evaluasi
│
├── test_images/           Folder untuk gambar pengujian manual
├── hand_landmarker.task   Model MediaPipe (diunduh otomatis jika belum ada)
└── requirements.txt       Daftar dependensi Python
```

---

## Cara Penggunaan

### 1. Instalasi Dependensi

```bash
pip install -r requirements.txt
```

### 2. Kumpulkan Dataset

```bash
python collect_dataset.py
```

Gunakan tombol `0`–`9` untuk memilih kelas, lalu `SPASI` untuk mengambil sampel atau `R` untuk rekam otomatis. Disarankan minimal **150 sampel per kelas** (total 1.500 sampel).

### 3. Latih Model

```bash
python train_model.py
```

Model terbaik disimpan secara otomatis berdasarkan validation loss terendah.

### 4. Uji dengan Gambar

```bash
python test_image.py <path_gambar.jpg>
```

### 5. Uji Real-Time via Kamera

```bash
python test_realtime.py
```

---

## Hyperparameter

| Parameter | Nilai | Keterangan |
|---|---|---|
| Learning Rate | 0.001 | Laju pembelajaran awal |
| Momentum | 0.9 | Koefisien SGD Momentum |
| Batch Size | 32 | Ukuran mini-batch |
| Max Epochs | 600 | Batas epoch training |
| LR Decay | 0.999 | Faktor peluruhan per epoch |
| Early Stopping | 80 | Toleransi epoch tanpa perbaikan |
| L2 Lambda | 0.0003 | Bobot regularization |
| Dropout Rate | 0.15 | Proporsi neuron dinonaktifkan saat training |

---

## Dependensi

| Library | Versi Minimum | Fungsi |
|---|---|---|
| numpy | 1.21.0 | Komputasi matriks JST |
| opencv-python | 4.5.0 | Akses kamera dan tampilan |
| mediapipe | 0.10.0 | Deteksi landmark tangan |
| matplotlib | 3.4.0 | Visualisasi grafik training |
| pyttsx3 | 2.90 | Text-to-speech (opsional) |

---

## Referensi

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV 2015*.
- Google MediaPipe Team. (2023). MediaPipe Hand Landmark Detection. https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik. Bebas digunakan dan dimodifikasi dengan menyertakan atribusi.
