"""
Utilitas Sistem Deteksi Hand Sign

Modul ini berisi semua fungsi pendukung yang digunakan oleh modul utama,
mencakup deteksi tangan via MediaPipe, ekstraksi dan normalisasi fitur
landmark, operasi dataset, normalisasi data, dan engine text-to-speech.

Modul ini dirancang agar kompatibel dengan dua versi MediaPipe:
  - Versi lama (< 0.10.14) menggunakan mp.solutions.hands
  - Versi baru (>= 0.10.14) menggunakan mediapipe.tasks.vision.HandLandmarker
"""

import numpy as np
import csv
import os
import threading

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    _MP_HAS_SOLUTIONS = hasattr(mp, 'solutions')
except ImportError:
    MP_AVAILABLE = False
    _MP_HAS_SOLUTIONS = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

import config


# Wrapper Hasil Deteksi MediaPipe

class _DetectionResult:
    """Menyatukan format hasil deteksi dari versi lama dan baru MediaPipe."""
    def __init__(self, multi_hand_landmarks=None, multi_handedness=None):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness     = multi_handedness


class _LandmarkPoint:
    """Merepresentasikan satu titik landmark tangan dengan koordinat x, y, z."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _LandmarkList:
    """Membungkus daftar landmark agar atribut .landmark bisa diakses seragam."""
    def __init__(self, landmark_list):
        self.landmark = [_LandmarkPoint(lm.x, lm.y, lm.z) for lm in landmark_list]


# Hand Detector (Kompatibel Dua Versi MediaPipe)

class HandDetector:
    """
    Wrapper untuk MediaPipe Hand Detection.

    Secara otomatis mendeteksi versi MediaPipe yang terinstal dan
    menggunakan API yang sesuai. Antarmuka deteksi dan gambar landmark
    dibuat seragam sehingga kode pemanggil tidak perlu tahu versi mana
    yang digunakan.
    """

    def __init__(self, static_image_mode=False):
        if not MP_AVAILABLE:
            raise ImportError("mediapipe belum terinstal. Jalankan: pip install mediapipe")

        self._use_legacy  = _MP_HAS_SOLUTIONS
        self._static_mode = static_image_mode

        if self._use_legacy:
            # API lama: mp.solutions.hands
            self.mp_hands         = mp.solutions.hands
            self.mp_drawing       = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
            )
        else:
            # API baru: mediapipe.tasks.vision.HandLandmarker
            import urllib.request
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "hand_landmarker.task"
            )

            if not os.path.isfile(model_path):
                print("[Info] Mengunduh model hand_landmarker.task (~8 MB)...")
                url = ("https://storage.googleapis.com/mediapipe-models/"
                       "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
                try:
                    urllib.request.urlretrieve(url, model_path)
                    print("[OK] Model berhasil diunduh.")
                except Exception as e:
                    raise RuntimeError(
                        f"Gagal mengunduh model MediaPipe: {e}\n"
                        "Unduh manual dari:\n"
                        "  https://storage.googleapis.com/mediapipe-models/"
                        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
                        f"Simpan sebagai: {model_path}"
                    )

            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            running_mode = (
                mp_vision.RunningMode.IMAGE
                if static_image_mode
                else mp_vision.RunningMode.VIDEO
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_hands=config.MAX_NUM_HANDS,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._frame_ts   = 0    # Timestamp dalam milidetik untuk mode VIDEO

    def detect(self, frame):
        """
        Mendeteksi tangan pada satu frame gambar.

        Mengembalikan objek _DetectionResult yang formatnya konsisten
        terlepas dari versi MediaPipe yang digunakan.
        """
        if self._use_legacy:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            return self.hands.process(rgb)
        else:
            from mediapipe.tasks.python import vision as mp_vision
            import mediapipe as _mp

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)

            if self._static_mode:
                detection = self._landmarker.detect(mp_image)
            else:
                self._frame_ts += 33    # Simulasi ~30 FPS
                detection = self._landmarker.detect_for_video(mp_image, self._frame_ts)

            if not detection.hand_landmarks:
                return _DetectionResult(multi_hand_landmarks=None)

            wrapped = [_LandmarkList(lm_list) for lm_list in detection.hand_landmarks]
            return _DetectionResult(multi_hand_landmarks=wrapped)

    def draw_landmarks(self, frame, results):
        """Menggambar titik dan koneksi landmark tangan pada frame."""
        if not results.multi_hand_landmarks:
            return frame

        if self._use_legacy:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        else:
            # Gambar manual dengan OpenCV untuk API versi baru
            h, w = frame.shape[:2]
            CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]
            for hand_lm in results.multi_hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm.landmark]
                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)
                    cv2.circle(frame, pt, 2, (0, 150, 80), -1)

        return frame

    def get_hand_label(self, results):
        """Mengembalikan label tangan yang terdeteksi (Left atau Right)."""
        if hasattr(results, 'multi_handedness') and results.multi_handedness:
            return results.multi_handedness[0].classification[0].label
        return None

    def close(self):
        """Melepaskan resource MediaPipe setelah selesai digunakan."""
        if self._use_legacy:
            self.hands.close()
        else:
            self._landmarker.close()


# Ekstraksi dan Normalisasi Fitur Landmark

def _angle_between(v1, v2):
    """
    Menghitung sudut dalam radian antara dua vektor 3D.

    Mengembalikan nilai antara 0 hingga pi. Jika salah satu vektor
    memiliki panjang mendekati nol, dikembalikan 0.
    """
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def extract_landmarks(hand_landmarks):
    """
    Mengubah 21 titik landmark tangan menjadi vektor fitur 86 dimensi.

    Proses normalisasi:
        1. Translasi: set pergelangan tangan (landmark 0) sebagai titik asal
        2. Skalasi: bagi dengan jarak pergelangan ke jari tengah MCP (lm 9)

    Komposisi fitur (total 86):
        63  koordinat ternormalisasi (21 titik x x,y,z)
        5   sudut tekukan tiap jari
        4   jarak ujung jari ke ibu jari
        5   status biner lurus/terlipat (threshold 0.65 rad)
        5   status biner jari ke atas (tip_y < mcp_y)
        4   jarak antar ujung jari yang berdekatan

    Parameters
    ----------
    hand_landmarks : MediaPipe hand landmarks object

    Returns
    -------
    np.ndarray, shape (86,)
    """
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Normalisasi posisi dan skala
    wrist     = landmarks[0].copy()
    landmarks = landmarks - wrist
    scale     = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks = landmarks / scale

    # Pasangan sendi untuk sudut tekukan per jari
    finger_joints = [
        (1, 2, 4),    # ibu jari
        (5, 6, 8),    # telunjuk
        (9, 10, 12),  # jari tengah
        (13, 14, 16), # jari manis
        (17, 18, 20), # kelingking
    ]

    bend_angles = []
    for base, mid, tip in finger_joints:
        v1 = landmarks[mid] - landmarks[base]
        v2 = landmarks[tip] - landmarks[mid]
        bend_angles.append(_angle_between(v1, v2))
    bend_angles = np.array(bend_angles)

    # Jarak ujung jari ke ibu jari
    thumb_tip     = landmarks[4]
    tip_distances = np.array([
        np.linalg.norm(landmarks[t] - thumb_tip) for t in [8, 12, 16, 20]
    ])

    # Status biner: apakah jari lurus (sudut tekukan < 0.65 radian)
    BEND_THRESHOLD = 0.65
    is_extended = (bend_angles < BEND_THRESHOLD).astype(float)

    # Status biner: apakah ujung jari lebih tinggi dari pangkal MCP
    tip_indices = [4, 8, 12, 16, 20]
    mcp_indices = [2, 5,  9, 13, 17]
    is_up = np.array([
        1.0 if landmarks[t][1] < landmarks[m][1] else 0.0
        for t, m in zip(tip_indices, mcp_indices)
    ])

    # Jarak antar ujung jari yang berdekatan
    tips     = [landmarks[i] for i in tip_indices]
    adj_dists = np.array([np.linalg.norm(tips[i+1] - tips[i]) for i in range(4)])

    return np.concatenate([
        landmarks.flatten(),    # 63 fitur
        bend_angles,            # 5 fitur
        tip_distances,          # 4 fitur
        is_extended,            # 5 fitur
        is_up,                  # 5 fitur
        adj_dists               # 4 fitur
    ])


def extract_landmarks_from_image(image_path):
    """
    Mengekstrak landmark dari file gambar statis.

    Returns
    -------
    tuple : (landmarks_array, annotated_image)
        landmarks_array adalah None jika tangan tidak terdeteksi.
    """
    if not CV2_AVAILABLE or not MP_AVAILABLE:
        raise ImportError("OpenCV dan MediaPipe diperlukan untuk fungsi ini")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    detector  = HandDetector(static_image_mode=True)
    results   = detector.detect(image)
    annotated = image.copy()

    if results.multi_hand_landmarks:
        landmarks = extract_landmarks(results.multi_hand_landmarks[0])
        annotated = detector.draw_landmarks(annotated, results)
        detector.close()
        return landmarks, annotated

    detector.close()
    return None, annotated


# Operasi Dataset (CSV)

def save_landmark_to_csv(landmarks, label, csv_path=None):
    """
    Menyimpan satu sampel landmark ke file CSV.

    Jika file belum ada, header akan dibuat secara otomatis.
    Setiap baris berisi label diikuti nilai 86 fitur.
    """
    if csv_path is None:
        csv_path = config.DATASET_CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
            writer.writerow(header)
        writer.writerow([label] + landmarks.tolist())


def load_dataset(csv_path=None):
    """
    Memuat dataset dari file CSV dan menghitung seluruh fitur turunan.

    Fungsi ini mendukung dua format CSV:
        Format lama : 63 kolom (21 landmark x x,y,z saja)
        Format baru : 86 kolom (63 koordinat + 23 fitur turunan)

    Untuk format lama, fitur turunan dihitung ulang secara otomatis
    agar konsisten dengan pipeline training terkini.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 86)
    y : np.ndarray, shape (n_samples,)  integer labels
    """
    if csv_path is None:
        csv_path = config.DATASET_CSV

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")

    finger_joints = [
        (1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20),
    ]
    other_tips  = [8, 12, 16, 20]
    tip_indices = [4, 8, 12, 16, 20]
    mcp_indices = [2, 5,  9, 13, 17]
    BEND_THRESHOLD = 0.65

    data, labels, skipped = [], [], 0

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)    # Lewati baris header
        for row in reader:
            raw = row[1:]

            if len(raw) == 63:
                coords = np.array([float(x) for x in raw])
            elif len(raw) >= 86:
                coords = np.array([float(x) for x in raw[:63]])
            else:
                skipped += 1
                continue

            labels.append(int(row[0]))
            lm = coords.reshape(21, 3)

            bend_angles = []
            for base, mid, tip in finger_joints:
                v1 = lm[mid] - lm[base]
                v2 = lm[tip] - lm[mid]
                bend_angles.append(_angle_between(v1, v2))

            thumb_tip = lm[4]
            tip_dists = [np.linalg.norm(lm[t] - thumb_tip) for t in other_tips]
            is_extended = [1.0 if a < BEND_THRESHOLD else 0.0 for a in bend_angles]
            is_up = [
                1.0 if lm[t][1] < lm[m][1] else 0.0
                for t, m in zip(tip_indices, mcp_indices)
            ]
            tips_pos  = [lm[i] for i in tip_indices]
            adj_dists = [np.linalg.norm(tips_pos[i+1] - tips_pos[i]) for i in range(4)]

            full_features = np.concatenate([
                coords,
                np.array(bend_angles),
                np.array(tip_dists),
                np.array(is_extended),
                np.array(is_up),
                np.array(adj_dists)
            ])
            data.append(full_features)

    X = np.array(data)
    y = np.array(labels)

    if skipped:
        print(f"[!] {skipped} baris dilewati (jumlah kolom tidak valid)")

    print(f"[OK] Dataset dimuat: {X.shape[0]} sampel, {X.shape[1]} fitur")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"     Angka {label}: {count} sampel")

    return X, y


# One-Hot Encoding dan Split Dataset

def one_hot_encode(y, num_classes=None):
    """Mengubah label integer menjadi representasi one-hot encoding."""
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
    return one_hot


def split_dataset(X, y, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    Membagi dataset menjadi train, validasi, dan test menggunakan Stratified Sampling.

    Stratified sampling memastikan setiap kelas memiliki proporsi yang
    seimbang di ketiga split, sehingga evaluasi model tidak bias.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(seed)

    y_int = np.argmax(y, axis=1) if y.ndim == 2 else y.astype(int)
    classes = np.unique(y_int)
    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        cls_indices = np.where(y_int == cls)[0]
        np.random.shuffle(cls_indices)
        n_cls  = len(cls_indices)
        n_test = max(1, int(n_cls * test_ratio))
        n_val  = max(1, int(n_cls * val_ratio))

        test_idx.extend(cls_indices[:n_test].tolist())
        val_idx.extend(cls_indices[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_indices[n_test + n_val:].tolist())

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])


# Normalisasi Data (Standard Scaler Manual

class ManualScaler:
    """
    Implementasi Standard Scaler (z-score normalization) tanpa scikit-learn.

    Menormalisasi setiap fitur agar memiliki rata-rata 0 dan standar deviasi 1.
    Parameter fit disimpan dari data training dan diterapkan ke data lainnya
    agar tidak terjadi data leakage.
    """

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, X):
        """Hitung rata-rata dan standar deviasi dari data training."""
        self.mean = np.mean(X, axis=0)
        self.std  = np.std(X, axis=0)
        self.std[self.std == 0] = 1     # Cegah pembagian dengan nol
        return self

    def transform(self, X):
        """Normalisasi data menggunakan parameter yang sudah dihitung."""
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """Hitung parameter dan normalisasi sekaligus."""
        self.fit(X)
        return self.transform(X)

    def save(self, filepath):
        """Simpan parameter scaler ke file .npz."""
        np.savez(filepath, mean=self.mean, std=self.std)
        print(f"[OK] Scaler disimpan ke: {filepath}")

    def load(self, filepath):
        """Muat parameter scaler dari file .npz."""
        data      = np.load(filepath)
        self.mean = data['mean']
        self.std  = data['std']
        print(f"[OK] Scaler dimuat dari: {filepath}")
        return self


# Text-to-Speech Engine

class VoiceEngine:
    """
    Engine text-to-speech untuk mengucapkan angka yang terdeteksi.

    Setiap ucapan dijalankan di thread terpisah dengan instance pyttsx3
    yang baru. Pendekatan ini menghindari bug pada Windows di mana
    pyttsx3.runAndWait() bisa macet (hang) jika engine digunakan ulang.

    Cooldown dikelola sepenuhnya di dalam kelas ini. Kode pemanggil
    hanya perlu memanggil speak() atau speak_number() tanpa perlu
    mengatur timer sendiri.
    """

    def __init__(self, enabled=True):
        self.enabled       = enabled and TTS_AVAILABLE
        self._last_spoken  = ""
        self._last_time    = 0.0
        self._voice_id     = None

        if self.enabled:
            try:
                eng    = pyttsx3.init()
                voices = eng.getProperty('voices')
                for v in voices:
                    if 'indonesia' in v.name.lower() or 'id' in v.id.lower():
                        self._voice_id = v.id
                        break
                eng.stop()
                del eng
                print("[OK] Text-to-Speech engine siap")
            except Exception as e:
                print(f"[!] TTS gagal diinisialisasi: {e}")
                self.enabled = False
        else:
            if not TTS_AVAILABLE:
                print("[!] pyttsx3 tidak tersedia. Install: pip install pyttsx3")

    def reset(self):
        """
        Mereset cooldown agar ucapan berikutnya bisa langsung muncul.
        Dipanggil saat gesture berubah atau tangan hilang dari kamera.
        """
        self._last_spoken = ""
        self._last_time   = 0.0

    def speak(self, text, force=False):
        """
        Mengucapkan teks secara non-blocking di thread terpisah.

        Pengucapan dilewati jika teks sama dengan yang terakhir diucapkan
        dan belum melewati periode cooldown, kecuali force=True.
        """
        import time as _time
        current_time = _time.time()

        if not self.enabled:
            return

        if not force:
            same_text      = (text == self._last_spoken)
            within_cooldown = (current_time - self._last_time) < config.VOICE_COOLDOWN
            if same_text and within_cooldown:
                return

        self._last_spoken = text
        self._last_time   = current_time

        voice_id = self._voice_id
        rate     = config.VOICE_RATE
        volume   = config.VOICE_VOLUME

        def _speak_thread():
            """Buat engine pyttsx3 baru per ucapan untuk menghindari hang di Windows."""
            try:
                eng = pyttsx3.init()
                eng.setProperty('rate', rate)
                eng.setProperty('volume', volume)
                if voice_id:
                    eng.setProperty('voice', voice_id)
                eng.say(text)
                eng.runAndWait()
                eng.stop()
            except Exception:
                pass    # Jangan sampai error di thread membuat aplikasi utama crash

        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()

    def speak_number(self, number):
        """Mengucapkan angka yang terdeteksi dalam Bahasa Indonesia."""
        text_map = {
            0: "nol", 1: "satu",  2: "dua",    3: "tiga",    4: "empat",
            5: "lima", 6: "enam", 7: "tujuh", 8: "delapan", 9: "sembilan"
        }
        self.speak(f"Angka {text_map.get(number, str(number))}")


# Statistik Dataset

def get_dataset_stats(csv_path=None):
    """
    Menghitung jumlah sampel per kelas dari file CSV dataset.

    Mengembalikan dictionary {label: jumlah_sampel} untuk semua kelas
    (0 hingga 9). Kelas yang belum memiliki sampel bernilai 0.
    """
    if csv_path is None:
        csv_path = config.DATASET_CSV

    if not os.path.isfile(csv_path):
        return {i: 0 for i in range(config.NUM_CLASSES)}

    counts = {i: 0 for i in range(config.NUM_CLASSES)}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                label = int(row[0])
                counts[label] = counts.get(label, 0) + 1

    return counts
