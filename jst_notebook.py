"""
============================================================================
JST BACKPROPAGATION - DETEKSI HAND SIGN ANGKA 0-9
VERSI GOOGLE COLAB / KAGGLE NOTEBOOK
============================================================================
File ini menggabungkan semua modul untuk dijalankan di Colab/Kaggle.

Cara pakai di Colab:
  1. Upload file ini ke Colab
  2. Upload dataset CSV (landmarks.csv) yang sudah direkam di lokal
  3. Jalankan cell-cell di bawah secara berurutan

Catatan:
  - Pengumpulan dataset (collect_dataset.py) harus dilakukan di lokal
    karena membutuhkan akses webcam secara real-time
  - Di Colab/Kaggle, fokus pada TRAINING dan TESTING
============================================================================
"""

# ============================================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================================
# !pip install mediapipe opencv-python-headless numpy matplotlib

import numpy as np
import os
import csv
import time
import json

try:
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# Deteksi environment
IN_COLAB = False
try:
    # pyrefly: ignore [missing-import]
    from google.colab import files as colab_files
    IN_COLAB = True
    print("[✓] Berjalan di Google Colab")
except ImportError:
    pass

IN_KAGGLE = os.path.exists('/kaggle')
if IN_KAGGLE:
    print("[✓] Berjalan di Kaggle")

if not IN_COLAB and not IN_KAGGLE:
    print("[i] Berjalan di environment lokal")


# ============================================================================
# CELL 2: KONFIGURASI
# ============================================================================

class Config:
    """Konfigurasi sistem."""
    # Path
    BASE_DIR = os.getcwd()
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATASET_CSV = os.path.join(DATASET_DIR, "landmarks.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "jst_hand_sign_model.npz")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.npz")

    # JST
    NUM_CLASSES = 10
    INPUT_SIZE = 63
    HIDDEN_LAYERS = [128, 64]
    OUTPUT_SIZE = NUM_CLASSES
    NETWORK_ARCHITECTURE = [INPUT_SIZE] + HIDDEN_LAYERS + [OUTPUT_SIZE]

    # Training
    LEARNING_RATE = 0.001
    EPOCHS = 500
    BATCH_SIZE = 32
    MOMENTUM = 0.9
    LEARNING_RATE_DECAY = 0.999
    EARLY_STOPPING_PATIENCE = 50
    REGULARIZATION_LAMBDA = 0.0001
    TEST_SPLIT = 0.2
    VALIDATION_SPLIT = 0.1
    RANDOM_SEED = 42
    MIN_SAMPLES_PER_CLASS = 30
    PREDICTION_CONFIDENCE_THRESHOLD = 0.6

cfg = Config()
os.makedirs(cfg.DATASET_DIR, exist_ok=True)
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
print(f"[✓] Konfigurasi dimuat. Arsitektur: {cfg.NETWORK_ARCHITECTURE}")


# ============================================================================
# CELL 3: IMPLEMENTASI JST BACKPROPAGATION
# ============================================================================

class NeuralNetwork:
    """
    Jaringan Syaraf Tiruan dengan Backpropagation (dari scratch).

    Arsitektur: [63] → [128 ReLU] → [64 ReLU] → [10 Softmax]
    Optimasi: Mini-Batch SGD + Momentum + L2 Regularization
    """

    def __init__(self, layer_sizes, learning_rate=0.001, momentum=0.9,
                 reg_lambda=0.0001, lr_decay=0.999):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.reg_lambda = reg_lambda
        self.lr_decay = lr_decay

        self.weights = []
        self.biases = []
        self._initialize_weights()

        self.vel_w = [np.zeros_like(w) for w in self.weights]
        self.vel_b = [np.zeros_like(b) for b in self.biases]
        self.activations = []
        self.z_values = []
        self.history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }

    def _initialize_weights(self):
        """He Initialization: W ~ N(0, sqrt(2/n_in))"""
        np.random.seed(42)
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """Forward propagation."""
        self.activations = [X]
        self.z_values = []
        a = X
        for i in range(self.n_layers - 2):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        a = self.softmax(z)
        self.activations.append(a)
        return a

    def cross_entropy_loss(self, y_pred, y_true):
        """Cross-Entropy Loss + L2 Regularization."""
        m = y_true.shape[0]
        epsilon = 1e-8
        ce_loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        l2_loss = sum(np.sum(w ** 2) for w in self.weights)
        l2_loss = (self.reg_lambda / (2 * m)) * l2_loss
        return ce_loss + l2_loss

    def backward(self, y_true):
        """Backpropagation - hitung gradien dan update bobot."""
        m = y_true.shape[0]
        delta = self.activations[-1] - y_true
        deltas = [delta]
        for i in range(self.n_layers - 3, -1, -1):
            delta = (deltas[-1] @ self.weights[i + 1].T) * \
                    self.relu_derivative(self.z_values[i])
            deltas.append(delta)
        deltas.reverse()
        for i in range(self.n_layers - 1):
            dW = (self.activations[i].T @ deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            dW += (self.reg_lambda / m) * self.weights[i]
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.learning_rate * dW
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.learning_rate * db
            self.weights[i] += self.vel_w[i]
            self.biases[i] += self.vel_b[i]

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=500, batch_size=32, patience=50, verbose=True):
        """Training dengan mini-batch SGD + Early Stopping."""
        m = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        self.history = {'train_loss': [], 'train_accuracy': [],
                        'val_loss': [], 'val_accuracy': []}

        if verbose:
            print("=" * 65)
            print("TRAINING JST BACKPROPAGATION")
            print(f"Arsitektur: {self.layer_sizes} | LR: {self.learning_rate} | "
                  f"Batch: {batch_size}")
            print("=" * 65)

        start = time.time()
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_s = X_train[indices]
            y_s = y_train[indices]

            for j in range(0, m, batch_size):
                self.forward(X_s[j:j + batch_size])
                self.backward(y_s[j:j + batch_size])

            self.learning_rate = self.initial_lr * (self.lr_decay ** epoch)

            y_p = self.forward(X_train)
            t_loss = self.cross_entropy_loss(y_p, y_train)
            t_acc = np.mean(np.argmax(y_p, axis=1) == np.argmax(y_train, axis=1))
            self.history['train_loss'].append(t_loss)
            self.history['train_accuracy'].append(t_acc)

            if X_val is not None:
                y_pv = self.forward(X_val)
                v_loss = self.cross_entropy_loss(y_pv, y_val)
                v_acc = np.mean(np.argmax(y_pv, axis=1) == np.argmax(y_val, axis=1))
                self.history['val_loss'].append(v_loss)
                self.history['val_accuracy'].append(v_acc)
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
                          f"Val Loss: {v_loss:.4f} Val Acc: {v_acc:.4f}")
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n[Early Stopping] Epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"Loss: {t_loss:.4f} Acc: {t_acc:.4f}")

        if best_weights:
            self.weights = best_weights
            self.biases = best_biases
        print(f"\nTraining selesai dalam {time.time()-start:.1f}s")
        return self.history

    def predict(self, X):
        return self.forward(X)

    def predict_class(self, X):
        return np.argmax(self.forward(X), axis=1)

    def predict_single(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self.forward(x)
        pc = np.argmax(probs, axis=1)[0]
        return pc, probs[0, pc], probs[0]

    def evaluate(self, X, y_true):
        y_pred = self.forward(X)
        loss = self.cross_entropy_loss(y_pred, y_true)
        pc = np.argmax(y_pred, axis=1)
        tc = np.argmax(y_true, axis=1)
        acc = np.mean(pc == tc)
        n = y_true.shape[1]
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(tc, pc):
            cm[t, p] += 1
        pa = {}
        for c in range(n):
            total = np.sum(tc == c)
            pa[c] = np.sum((tc == c) & (pc == c)) / total if total > 0 else 0
        return {'accuracy': acc, 'loss': loss, 'confusion_matrix': cm,
                'per_class_accuracy': pa}

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        d = {'layer_sizes': np.array(self.layer_sizes),
             'learning_rate': np.array([self.initial_lr]),
             'momentum': np.array([self.momentum]),
             'reg_lambda': np.array([self.reg_lambda])}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            d[f'weight_{i}'] = w
            d[f'bias_{i}'] = b
        np.savez(filepath, **d)
        print(f"[✓] Model disimpan: {filepath}")

    @classmethod
    def load_model(cls, filepath):
        data = np.load(filepath)
        ls = data['layer_sizes'].tolist()
        m = cls(ls, float(data['learning_rate'][0]),
                float(data['momentum'][0]), float(data['reg_lambda'][0]))
        nl = len(ls) - 1
        m.weights = [data[f'weight_{i}'] for i in range(nl)]
        m.biases = [data[f'bias_{i}'] for i in range(nl)]
        m.vel_w = [np.zeros_like(w) for w in m.weights]
        m.vel_b = [np.zeros_like(b) for b in m.biases]
        print(f"[✓] Model dimuat: {ls}")
        return m

    def summary(self):
        print("\n" + "=" * 50)
        print("RINGKASAN MODEL JST BACKPROPAGATION")
        print("=" * 50)
        total = 0
        for i in range(self.n_layers - 1):
            p = self.weights[i].size + self.biases[i].size
            total += p
            act = "ReLU" if i < self.n_layers - 2 else "Softmax"
            print(f"Layer {i}: {self.layer_sizes[i]:4d} → {self.layer_sizes[i+1]:4d} "
                  f"({act:7s}) | Params: {p:,}")
        print(f"Total Parameters: {total:,}")
        print("=" * 50)


# ============================================================================
# CELL 4: UTILITAS
# ============================================================================

class ManualScaler:
    """Standard Scaler manual."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, filepath):
        np.savez(filepath, mean=self.mean, std=self.std)

    def load(self, filepath):
        data = np.load(filepath)
        self.mean = data['mean']
        self.std = data['std']
        return self


def one_hot_encode(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes))
    oh[np.arange(y.shape[0]), y.astype(int)] = 1
    return oh


def split_dataset(X, y, test_ratio=0.2, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    ts = int(n * test_ratio)
    vs = int(n * val_ratio)
    return (X[idx[ts+vs:]], y[idx[ts+vs:]],
            X[idx[ts:ts+vs]], y[idx[ts:ts+vs]],
            X[idx[:ts]], y[idx[:ts]])


def load_dataset(csv_path):
    """Load dataset dari CSV."""
    data, labels = [], []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            labels.append(int(row[0]))
            data.append([float(x) for x in row[1:]])
    X = np.array(data)
    y = np.array(labels)
    print(f"[✓] Dataset: {X.shape[0]} sampel, {X.shape[1]} fitur")
    unique, counts = np.unique(y, return_counts=True)
    for l, c in zip(unique, counts):
        print(f"    Angka {l}: {c} sampel")
    return X, y


def extract_landmarks(hand_landmarks):
    """Ekstrak dan normalisasi 21 landmark → 63 fitur."""
    lms = []
    for lm in hand_landmarks.landmark:
        lms.append([lm.x, lm.y, lm.z])
    lms = np.array(lms)
    wrist = lms[0].copy()
    lms = lms - wrist
    scale = np.linalg.norm(lms[9])
    if scale > 0:
        lms = lms / scale
    return lms.flatten()


# ============================================================================
# CELL 5: UPLOAD & LOAD DATASET (untuk Colab/Kaggle)
# ============================================================================

def upload_and_load_dataset():
    """Upload CSV dataset di Colab, atau load dari path."""
    csv_path = cfg.DATASET_CSV

    if IN_COLAB:
        print("Upload file landmarks.csv yang telah direkam di lokal:")
        uploaded = colab_files.upload()
        for name, content in uploaded.items():
            csv_path = os.path.join(cfg.DATASET_DIR, name)
            with open(csv_path, 'wb') as f:
                f.write(content)
            print(f"[✓] File disimpan: {csv_path}")
            break
    elif IN_KAGGLE:
        # Di Kaggle, pastikan dataset sudah di-add
        possible = ['/kaggle/input/hand-sign-dataset/landmarks.csv',
                    '/kaggle/input/landmarks.csv',
                    cfg.DATASET_CSV]
        for p in possible:
            if os.path.isfile(p):
                csv_path = p
                break
        print(f"[i] Menggunakan dataset: {csv_path}")
    else:
        if not os.path.isfile(csv_path):
            csv_path = input("Masukkan path ke landmarks.csv: ").strip()

    X, y = load_dataset(csv_path)
    return X, y


# ============================================================================
# CELL 6: TRAINING
# ============================================================================

def run_training(X, y):
    """Jalankan training JST."""
    print("\n" + "=" * 65)
    print("  TRAINING JST BACKPROPAGATION - HAND SIGN")
    print("=" * 65)

    # Preprocessing
    scaler = ManualScaler()
    X_scaled = scaler.fit_transform(X)
    y_oh = one_hot_encode(y, cfg.NUM_CLASSES)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_scaled, y_oh, cfg.TEST_SPLIT, cfg.VALIDATION_SPLIT, cfg.RANDOM_SEED)

    print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # Buat dan latih model
    model = NeuralNetwork(
        cfg.NETWORK_ARCHITECTURE, cfg.LEARNING_RATE,
        cfg.MOMENTUM, cfg.REGULARIZATION_LAMBDA, cfg.LEARNING_RATE_DECAY)
    model.summary()

    history = model.train(X_train, y_train, X_val, y_val,
                          cfg.EPOCHS, cfg.BATCH_SIZE, cfg.EARLY_STOPPING_PATIENCE)

    # Evaluasi
    print("\n--- Evaluasi Test Set ---")
    results = model.evaluate(X_test, y_test)
    print(f"Akurasi: {results['accuracy']*100:.2f}%")
    print(f"Loss: {results['loss']:.4f}")
    print("\nAkurasi per kelas:")
    for c, a in results['per_class_accuracy'].items():
        print(f"  Angka {c}: {a*100:.1f}%")

    # Simpan model
    model.save_model(cfg.MODEL_PATH)
    scaler.save(cfg.SCALER_PATH)

    # Plot
    if PLT_AVAILABLE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history['train_loss'], label='Train', color='#FF6B6B', lw=2)
        if history['val_loss']:
            ax1.plot(history['val_loss'], label='Val', color='#4ECDC4', lw=2)
        ax1.set_title('Loss per Epoch', fontweight='bold')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(history['train_accuracy'], label='Train', color='#FF6B6B', lw=2)
        if history['val_accuracy']:
            ax2.plot(history['val_accuracy'], label='Val', color='#4ECDC4', lw=2)
        ax2.set_title('Akurasi per Epoch', fontweight='bold')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
        ax2.grid(True, alpha=0.3); ax2.set_ylim([0, 1.05])
        plt.suptitle('Training History - JST Backpropagation', fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.show()

        # Confusion Matrix
        cm = results['confusion_matrix']
        fig2, ax3 = plt.subplots(figsize=(8, 6))
        im = ax3.imshow(cm, cmap='Blues')
        ax3.figure.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, str(cm[i,j]), ha='center', va='center',
                         color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax3.set_xlabel('Prediksi'); ax3.set_ylabel('Aktual')
        ax3.set_title('Confusion Matrix', fontweight='bold')
        ax3.set_xticks(range(10)); ax3.set_yticks(range(10))
        plt.tight_layout(); plt.show()

    return model, scaler, results


# ============================================================================
# CELL 7: TEST DENGAN GAMBAR (untuk Colab/Kaggle)
# ============================================================================

def test_with_image(model, scaler, image_path=None):
    """Test model dengan gambar hand sign."""
    if not CV2_AVAILABLE or not MP_AVAILABLE:
        print("[ERROR] OpenCV dan MediaPipe diperlukan")
        return

    if image_path is None:
        if IN_COLAB:
            print("Upload gambar hand sign:")
            uploaded = colab_files.upload()
            for name in uploaded:
                image_path = name
                break
        else:
            image_path = input("Path gambar: ").strip()

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Gagal membaca: {image_path}")
        return

    # Deteksi landmark
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=0.5)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        print("[!] Tangan tidak terdeteksi!")
        if PLT_AVAILABLE:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Tangan Tidak Terdeteksi"); plt.axis('off'); plt.show()
        return

    # Ekstrak dan prediksi
    lm = extract_landmarks(results.multi_hand_landmarks[0])
    lm_scaled = scaler.transform(lm.reshape(1, -1))
    pred, conf, probs = model.predict_single(lm_scaled)

    print(f"\n  Prediksi: Angka {pred} (Confidence: {conf*100:.1f}%)")

    # Visualisasi
    if PLT_AVAILABLE:
        # Gambar landmark
        annotated = img.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated, results.multi_hand_landmarks[0],
            mp.solutions.hands.HAND_CONNECTIONS)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax1.set_title("Hand Sign"); ax1.axis('off')

        colors = ['#4ECDC4' if i != pred else '#FF6B6B' for i in range(10)]
        ax2.barh(range(10), probs * 100, color=colors)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels([f'Angka {i}' for i in range(10)])
        ax2.set_xlabel('Probabilitas (%)')
        ax2.set_title(f'Prediksi: Angka {pred} ({conf*100:.1f}%)', fontweight='bold')
        plt.tight_layout(); plt.show()

    hands.close()


# ============================================================================
# CELL 8: MAIN - JALANKAN SEMUA
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  JST BACKPROPAGATION - DETEKSI HAND SIGN ANGKA 0-9")
    print("  Versi Notebook (Colab/Kaggle/Lokal)")
    print("=" * 65)
    print()
    print("Langkah-langkah:")
    print("  1. Upload dataset  : upload_and_load_dataset()")
    print("  2. Training        : run_training(X, y)")
    print("  3. Test gambar     : test_with_image(model, scaler)")
    print()
    print("Contoh penggunaan:")
    print("  X, y = upload_and_load_dataset()")
    print("  model, scaler, results = run_training(X, y)")
    print("  test_with_image(model, scaler, 'test.jpg')")
    print()

    # Untuk running langsung:
    # X, y = upload_and_load_dataset()
    # model, scaler, results = run_training(X, y)
    # test_with_image(model, scaler)
